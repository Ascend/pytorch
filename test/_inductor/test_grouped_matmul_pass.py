import operator
import unittest

import torch
from torch.export import Dim, export
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from testutils import TestUtils
import torch_npu  # noqa: F401
from torch_npu._inductor import config as npu_config
from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import (
    _gmm_plan_batch_size,
    grouped_matmul_fusion_pass,
)


_CAT = torch.ops.aten.cat.default
_ADDMM = torch.ops.aten.addmm.default
_MM = torch.ops.aten.mm.default
_RELU = torch.ops.aten.relu.default
_GMM = torch.ops.npu.npu_grouped_matmul.default

_ROWS = 200
_WIDTH = 64
# Tower widths from a real model: three orders of magnitude, most far below a cube's capacity.
_TOWER_K = (16, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1024)

_ENABLED = npu_config.enable_grouped_matmul_fusion
_NEEDS_FLAG = "requires TORCHINDUCTOR_ENABLE_GROUPED_MATMUL_FUSION=1 (the pass registers on the switch)"


def _count(graph, target):
    return len([n for n in graph.nodes if n.op == "call_function" and n.target == target])


def _grouped_nodes(graph):
    return [n for n in graph.nodes if n.op == "call_function" and n.target == _GMM]


def _cat_node(graph):
    return [n for n in graph.nodes if n.op == "call_function" and n.target == _CAT][-1]


def _batch_sizes(graph):
    return [len(n.args[0]) for n in _grouped_nodes(graph)]


def _member_order(graph):
    """Map each cat input back to (batch index, group index) to confirm concat order is intact.

    After the rewrite every cat input is a getitem, so node counts cannot catch a permutation,
    which would silently interleave the features of different towers.
    """
    grouped = _grouped_nodes(graph)
    order = []
    for inp in _cat_node(graph).args[0]:
        if not (isinstance(inp, torch.fx.Node) and inp.target is operator.getitem):
            order.append(None)
            continue
        order.append((grouped.index(inp.args[0]), inp.args[1]))
    return order


class _Captured:
    """A captured graph plus a callable that runs it."""

    def __init__(self, gm, call):
        self.gm = gm
        self.graph = gm.graph
        self._call = call

    def __call__(self, *args):
        return self._call(*args)


class _FnModule(torch.nn.Module):
    """torch.export only accepts nn.Module, and the cases under test are functions.

    Inputs arrive as one tuple so dynamic_shapes stays a single positional spec no
    matter how many tensors a case passes; a tower of n projections passes 2n or 3n
    of them. Export flattens it back to one placeholder per tensor.
    """

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def forward(self, tensors):
        return self._fn(*tensors)


def _capture(fn, *tensors, dynamic=False):
    """Capture the graph the pass actually operates on.

    Static mode uses ``tracing_mode="fake"`` so all placeholders share one FakeTensorMode, as in
    production. Dynamic mode uses ``torch.export`` with only dim 0 dynamic, because batch is the
    only symbolic shape in production and each branch's K is a compile-time constant.
    """
    if not dynamic:
        gm = make_fx(fn, tracing_mode="fake")(*tensors)
        return _Captured(gm, gm)

    batch = Dim("batch", min=2)
    dynamic_shapes = tuple(
        {0: batch} if t.dim() == 2 and t.shape[0] == _ROWS else None for t in tensors
    )
    program = export(_FnModule(fn), (tuple(tensors),), dynamic_shapes=(dynamic_shapes,))
    return _Captured(program.graph_module, lambda *a: program.module()(tuple(a)))


class TestGroupedMatmulPass(TestUtils):
    def _tower_inputs(self, n, rows=_ROWS, width=_WIDTH, dtype="float16", bias=True):
        """Build n projections with distinct K but equal output width, plus the final cat inputs."""
        torch_dtype = eval("torch." + dtype)
        device = torch.device("npu")
        ks = [_TOWER_K[i % len(_TOWER_K)] for i in range(n)]
        xs = [torch.randn((rows, k), dtype=torch_dtype, device=device) for k in ks]
        ws = [torch.randn((k, width), dtype=torch_dtype, device=device) for k in ks]
        if not bias:
            return xs + ws
        bs = [torch.randn((width,), dtype=torch_dtype, device=device) for _ in ks]
        return xs + ws + bs

    def _tower_fn(self, n, bias=True):
        def fn(*tensors):
            xs, ws = tensors[:n], tensors[n:2 * n]
            if bias:
                bs = tensors[2 * n:3 * n]
                parts = [_ADDMM(b, x, w) for b, x, w in zip(bs, xs, ws)]
            else:
                parts = [_MM(x, w) for x, w in zip(xs, ws)]
            return _CAT(parts, 1)

        return fn

    def _run_pass(self, fn, *tensors, dynamic=False, **limits):
        captured = _capture(fn, *tensors, dynamic=dynamic)
        grouped_matmul_fusion_pass(captured.graph, **limits)
        captured.gm.recompile()
        return captured

    # ------------------------------------------------------------------
    # Batching rule. The cliff sits between 32 and 40 groups: minimize batches, then split evenly.
    # ------------------------------------------------------------------
    @parametrize(
        "total,expected",
        [
            (8, 8),      # fits in one batch
            (32, 32),    # exactly at the cap
            (33, 17),    # just over the cap: two even batches, not 32 + 1
            (64, 32),
            (80, 27),    # measured optimum: 3 batches of 27/27/26, 14% faster than 32/32/16
            (100, 25),
        ],
    )
    def test_batch_size_is_balanced(self, total, expected):
        self.assertEqual(_gmm_plan_batch_size(total, 32), expected)

    def test_batch_size_never_exceeds_cap(self):
        for cap in (8, 16, 32):
            for total in range(2, 200):
                self.assertLessEqual(_gmm_plan_batch_size(total, cap), cap,
                                     msg=f"cap={cap} total={total}")

    # ------------------------------------------------------------------
    # Graph rewriting
    # ------------------------------------------------------------------
    def test_folds_tower_into_balanced_batches(self):
        n = 80
        captured = self._run_pass(self._tower_fn(n), *self._tower_inputs(n),
                                  max_groups_per_call=32)
        self.assertEqual(_batch_sizes(captured.graph), [27, 27, 26])
        self.assertEqual(_count(captured.graph, _ADDMM), 0)
        # Dispatch count is the whole point of this rewrite: 81 (80 matmul + 1 cat) down to 4.
        self.assertEqual(_count(captured.graph, _GMM) + _count(captured.graph, _CAT), 4)

    def test_preserves_concat_order(self):
        n = 80
        captured = self._run_pass(self._tower_fn(n), *self._tower_inputs(n),
                                  max_groups_per_call=32)
        expected = ([(0, i) for i in range(27)]
                    + [(1, i) for i in range(27)]
                    + [(2, i) for i in range(26)])
        self.assertEqual(_member_order(captured.graph), expected)

    def test_passes_group_semantics(self):
        """split_item=0 emits one output per group; group_type=-1 gives each group its own K."""
        n = 16
        captured = self._run_pass(self._tower_fn(n), *self._tower_inputs(n))
        for node in _grouped_nodes(captured.graph):
            self.assertEqual(node.kwargs["split_item"], 0)
            self.assertEqual(node.kwargs["group_type"], -1)
            self.assertEqual(len(node.kwargs["bias"]), len(node.args[0]))
            self.assertEqual(len(node.args[0]), len(node.args[1]))

    def test_folds_without_bias(self):
        n = 16
        captured = self._run_pass(self._tower_fn(n, bias=False),
                                  *self._tower_inputs(n, bias=False))
        self.assertEqual(_count(captured.graph, _MM), 0)
        for node in _grouped_nodes(captured.graph):
            self.assertNotIn("bias", node.kwargs)

    def test_folds_under_dynamic_batch(self):
        """Row count is symbolic in production graphs; a symbolic dim must not block matching."""
        n = 16
        captured = self._run_pass(self._tower_fn(n), *self._tower_inputs(n),
                                  dynamic=True)
        self.assertEqual(_count(captured.graph, _GMM), 1)
        self.assertEqual(_count(captured.graph, _ADDMM), 0)

    def test_splits_biased_and_plain_into_separate_calls(self):
        """Biased and plain matmuls cannot share a batch: bias is a whole-call argument."""
        n = 20
        tensors = self._tower_inputs(n)

        def fn(*t):
            xs, ws, bs = t[:n], t[n:2 * n], t[2 * n:]
            parts = [_ADDMM(b, x, w) for b, x, w in zip(bs[:10], xs[:10], ws[:10])]
            parts += [_MM(x, w) for x, w in zip(xs[10:], ws[10:])]
            return _CAT(parts, 1)

        captured = self._run_pass(fn, *tensors, min_groups=8)
        self.assertEqual(_count(captured.graph, _GMM), 2)
        with_bias = [n_ for n_ in _grouped_nodes(captured.graph) if "bias" in n_.kwargs]
        self.assertEqual(len(with_bias), 1)

    # ------------------------------------------------------------------
    # Rejection conditions
    # ------------------------------------------------------------------
    def test_skips_few_branches(self):
        """Few wide GEMMs like QKV already fill the cube; grouping them costs 12% device time."""
        n = 3
        captured = self._run_pass(self._tower_fn(n), *self._tower_inputs(n),
                                  min_groups=8)
        self.assertEqual(_count(captured.graph, _GMM), 0)
        self.assertEqual(_count(captured.graph, _ADDMM), n)

    def test_skips_wide_rows_when_static(self):
        n = 16
        captured = self._run_pass(self._tower_fn(n),
                                  *self._tower_inputs(n, rows=_ROWS),
                                  max_rows=_ROWS - 1)
        self.assertEqual(_count(captured.graph, _GMM), 0)

    def test_skips_row_concat(self):
        n = 16
        tensors = self._tower_inputs(n, width=_WIDTH)

        def fn(*t):
            xs, ws, bs = t[:n], t[n:2 * n], t[2 * n:]
            return _CAT([_ADDMM(b, x, w) for b, x, w in zip(bs, xs, ws)], 0)

        captured = self._run_pass(fn, *tensors)
        self.assertEqual(_count(captured.graph, _GMM), 0)

    def test_skips_mismatched_width(self):
        """Differing output widths form separate buckets, neither meeting the minimum group."""
        n = 16
        device = torch.device("npu")
        xs = [torch.randn((_ROWS, 64), dtype=torch.float16, device=device)
              for _ in range(n)]
        ws = [torch.randn((64, _WIDTH if i < n // 2 else _WIDTH * 2),
                          dtype=torch.float16, device=device) for i in range(n)]
        bs = [torch.randn((_WIDTH if i < n // 2 else _WIDTH * 2,),
                          dtype=torch.float16, device=device) for i in range(n)]

        def fn(*t):
            a, b, c = t[:n], t[n:2 * n], t[2 * n:]
            return _CAT([_ADDMM(z, x, w) for z, x, w in zip(c, a, b)], 1)

        captured = self._run_pass(fn, *xs, *ws, *bs, min_groups=12)
        self.assertEqual(_count(captured.graph, _GMM), 0)

    def test_skips_scaled_addmm(self):
        """addmm with beta/alpha is not plain x @ w + b; the grouped operator cannot express it."""
        n = 16
        tensors = self._tower_inputs(n)

        def fn(*t):
            xs, ws, bs = t[:n], t[n:2 * n], t[2 * n:]
            parts = [_ADDMM(b, x, w, beta=2.0) for b, x, w in zip(bs, xs, ws)]
            return _CAT(parts, 1)

        captured = self._run_pass(fn, *tensors)
        self.assertEqual(_count(captured.graph, _GMM), 0)
        self.assertEqual(_count(captured.graph, _ADDMM), n)

    def test_skips_matmul_with_other_users(self):
        """Extra users force the grouped call before the earliest user, so the gain is uncertain."""
        n = 16
        tensors = self._tower_inputs(n)

        def fn(*t):
            xs, ws, bs = t[:n], t[n:2 * n], t[2 * n:]
            parts = [_ADDMM(b, x, w) for b, x, w in zip(bs, xs, ws)]
            return _CAT(parts, 1), _RELU(parts[0])

        captured = self._run_pass(fn, *tensors, min_groups=8)
        # Only the first branch has an extra user; the remaining 15 still meet the minimum.
        self.assertEqual(_batch_sizes(captured.graph), [15])
        self.assertEqual(_count(captured.graph, _ADDMM), 1)

    # ------------------------------------------------------------------
    # Numerics. Not bitwise-identical: the operator accumulates in a different order than separate
    # matmuls; measured relative RMS error in fp16 is 6.6e-06, matmul's own noise level.
    # ------------------------------------------------------------------
    @unittest.skipUnless(_ENABLED, _NEEDS_FLAG)
    @parametrize("n", [16, 80])
    def test_matches_separate_matmuls(self, n):
        tensors = self._tower_inputs(n)
        fn = self._tower_fn(n)
        expected = fn(*tensors)
        captured = self._run_pass(fn, *tensors,
                                  max_groups_per_call=32)
        self.assertGreater(_count(captured.graph, _GMM), 0, "pass did not fire, so this case proves nothing")
        actual = captured(*tensors)
        self.assertEqual(expected.shape, actual.shape)

        # Element-wise comparison is meaningless here: accumulating 1024 fp16 products puts one ULP
        # at 3e-2, so any element-wise threshold is either too loose to catch bugs or bound to
        # flake. Relative RMS error is stable: 6.6e-06 measured at harder tower widths (K to 5036).
        diff = (actual.float() - expected.float())
        rms = diff.pow(2).mean().sqrt() / expected.float().pow(2).mean().sqrt()
        self.assertLess(rms.item(), 1e-4, f"relative RMS error {rms.item():.3e} is too large")


instantiate_parametrized_tests(TestGroupedMatmulPass)


if __name__ == "__main__":
    run_tests()
