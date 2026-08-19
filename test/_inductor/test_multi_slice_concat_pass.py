import contextlib
import types
import unittest

import torch
from torch._inductor.utils import run_and_get_code
from torch.export import Dim, export
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from testutils import TestUtils
import torch_npu
import torch_npu._inductor
from torch_npu._inductor import config as npu_config
from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import (
    MULTI_SLICE_CONCAT_TARGET,
    multi_slice_concat_pass,
)
from torch_npu._inductor.kernel.multi_slice_concat import _dedup_inputs


_CAT = torch.ops.aten.cat.default
_SLICE = torch.ops.aten.slice.Tensor
_WHERE = torch.ops.aten.where.self
_FULL = torch.ops.aten.full.default
_MSC = torch.ops.npu_ext.multi_slice_concat

# operator arg positions, so node.args is not indexed by bare numbers
_A_SRCS, _A_MASKS, _A_SRC_IDX, _A_OFFSETS, _A_WIDTHS, _A_MASK_IDX = range(6)

# wide feature table and offsets from real model output code, all compile-time constants.
_WIDE_COLS = 69876
# L24837: 15 segments, width=128.
_W128_OFFSETS = (55834, 56452, 45421, 44203, 44744, 45962, 60741, 61328,
                 62267, 65602, 66495, 66956, 67445, 68021, 69060)
# L24895: 12 segments, width=16.
_W16_OFFSETS = (55818, 56436, 45405, 44187, 44728, 45946, 61312, 62251,
                69044, 68199, 68221, 69822)
# L25021: 45 mixed-width segments, the longest column concat in the model.
_MIXED_OFFSETS = (55818, 56436, 45405, 44187, 44728, 45946, 61312, 62251, 69044,
                  45421, 44203, 44744, 45962, 60741, 61328, 62267, 65602, 66495,
                  66956, 67445, 69060, 45549, 44331, 44872, 46090, 61456, 62395,
                  69188, 68199, 68221, 69822, 55580, 59302, 1693, 1749, 1807,
                  2033, 3276, 3350, 2463, 58205, 58980, 62964, 2159, 64802)
_MIXED_WIDTHS = (16, 16, 16, 16, 16, 16, 16, 16, 16, 128, 128, 128, 128, 128, 128,
                 128, 128, 128, 128, 128, 128, 32, 32, 32, 32, 32, 32, 32, 16, 16,
                 16, 128, 128, 32, 32, 32, 32, 32, 32, 32, 16, 16, 16, 16, 16)
# L27947: 28 segments, width=8, the narrowest ones in the model.
_W8_OFFSETS = (48025, 47595, 46868, 47890, 48779, 49346, 47769, 49082, 49180,
               48337, 48467, 48597, 48402, 48532, 48714, 49240, 48070, 48204,
               48119, 11076, 14257, 18643, 11922, 15103, 19489, 12752, 15933, 20319)
# cat_124: masked and bare slices strictly alternating; nothing merged before masks.
_ALT_OFFSETS = (35098, 39296, 35470, 39913, 34648, 38647, 35842, 40530)
_ALT_WIDTHS = (80, 80, 80, 80, 112, 112, 128, 128)

_ENABLED = npu_config.enable_multi_slice_concat
_NEEDS_FLAG = "needs TORCHINDUCTOR_ENABLE_MULTI_SLICE_CONCAT=1 (pass is registered by the flag)"


def _count(graph, target):
    return len([n for n in graph.nodes if n.op == "call_function" and n.target == target])


def _op_nodes(graph):
    return [n for n in graph.nodes
            if n.op == "call_function" and n.target == MULTI_SLICE_CONCAT_TARGET]


def _kernel_source(code, marker):
    """Source of the one compiled kernel whose body contains marker, or None.

    Every kernel in the output code is its own ``async_compile.triton()`` string, so
    splitting on that keeps kernels apart; requiring a def line drops the call section.
    Kernels are matched on a rendered argument name, not on the kernel name, which
    define_kernel builds from node origins rather than from the template.
    """
    for chunk in code.split("async_compile.triton("):
        if marker in chunk and any(
            line.startswith("def ") for line in chunk.splitlines()
        ):
            return chunk
    return None


@contextlib.contextmanager
def _pass_spy():
    """Count the op nodes the pass produces during compilation.

    When the pass does not fire the result falls back to aten.cat and is still
    bitwise correct, so a numeric-only end-to-end check passes vacuously. The
    pass is registered as a function object, so swap in a counting wrapper.
    """
    from torch_npu._inductor.fx_passes.ascend_custom_passes.register_custom_pass import (
        ASCEND_CUSTOME_PASS_REGISTER,
    )

    stats = {"nodes": 0}
    original = []

    def spy(graph):
        multi_slice_concat_pass(graph)
        stats["nodes"] += len(_op_nodes(graph))

    spy.__name__ = multi_slice_concat_pass.__name__
    for pass_type, levels in ASCEND_CUSTOME_PASS_REGISTER.items():
        for level, fns in levels.items():
            if any(getattr(f, "__name__", "") == spy.__name__ for f in fns):
                original.append((pass_type, level, list(fns)))
                levels[level] = [
                    spy if getattr(f, "__name__", "") == spy.__name__ else f
                    for f in fns
                ]
    try:
        yield stats
    finally:
        for pass_type, level, fns in original:
            ASCEND_CUSTOME_PASS_REGISTER[pass_type][level] = fns


class _Captured:
    def __init__(self, gm, call):
        self.gm = gm
        self.graph = gm.graph
        self._call = call

    def __call__(self, *args):
        return self._call(*args)


class _FnModule(torch.nn.Module):
    """torch.export only accepts nn.Module, and the cases under test are functions.

    Inputs arrive as one tuple so dynamic_shapes stays a single positional spec no
    matter how many tensors a case passes; export flattens it back to one
    placeholder per tensor, which is what the pass sees.
    """

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def forward(self, tensors):
        return self._fn(*tensors)


def _capture(fn, *tensors, dynamic=False):
    """Capture the graph the pass actually sees.

    Static uses ``tracing_mode="fake"``, not ``"real"``: real builds one
    FakeTensorMode per input, so inferring output meta for multi-source segments
    hits "Mixing fake modes" and silently skips the rewrite.

    Dynamic uses ``torch.export`` with only dim 0 dynamic, like production:
    symbolic batch, static last dim. ``tracing_mode="symbolic"`` would symbolize
    the last dim too, leaving the pass without constant offsets so it silently
    stops matching.
    """
    if not dynamic:
        gm = make_fx(fn, tracing_mode="fake")(*tensors)
        return _Captured(gm, gm)

    batch = Dim("batch", min=2)
    program = export(_FnModule(fn), (tuple(tensors),),
                     dynamic_shapes=(tuple({0: batch} for _ in tensors),))
    return _Captured(program.graph_module, lambda *a: program.module()(tuple(a)))


def _col_concat(wide, offsets, widths):
    """Reproduce the [aten.slice] -> cat(dim=-1) shape from the output code."""
    parts = [_SLICE(wide, -1, off, off + w) for off, w in zip(offsets, widths)]
    return _CAT(parts, -1)


def _col_concat_ref(wide, offsets, widths):
    return torch.cat([wide[..., off:off + w] for off, w in zip(offsets, widths)], dim=-1)


def _masked_slice(wide, mask, off, width):
    """Reproduce the where(row mask, zeros, column slice) shape from output code."""
    part = _SLICE(wide, -1, off, off + width)
    zero = _FULL([wide.shape[0], width], 0, dtype=wide.dtype, device=wide.device)
    return _WHERE(mask, zero, part)


def _masked_slice_ref(wide, mask, off, width):
    return torch.where(mask, torch.zeros_like(wide[..., off:off + width]),
                       wide[..., off:off + width])


class _StubInput:
    """_dedup_inputs only reads buffer name and layout, so no real IR nodes needed."""

    def __init__(self, name, size=(2, 1), stride=(1, 1), offset=0):
        self._name = name
        self._layout = types.SimpleNamespace(
            size=list(size), stride=list(stride), offset=offset
        )

    def get_name(self):
        return self._name

    def get_layout(self):
        return self._layout


class TestMultiSliceConcatPass(TestUtils):
    # pure data movement plus mask select, so output must be bitwise-identical to cat.
    def _assert_bitwise_equal(self, expected, actual, note=""):
        self.assertEqual(expected, actual, atol=0, rtol=0,
                         msg=f"{note}: a pure-copy rewrite must be bitwise identical")

    def _wide(self, rows, dtype='float16', cols=_WIDE_COLS):
        return torch.randn((rows, cols), dtype=eval('torch.' + dtype),
                           device=torch.device("npu"))

    def _mask(self, rows, cols=1):
        """Row-wise bool mask [rows, 1], matching logical_not in the model."""
        return torch.randint(0, 2, (rows, cols), device=torch.device("npu"),
                             dtype=torch.bool)

    def _run_pass(self, fn, *tensors, dynamic=False, **limits):
        captured = _capture(fn, *tensors, dynamic=dynamic)
        multi_slice_concat_pass(captured.graph, **limits)
        captured.gm.recompile()
        return captured

    # ------------------------------------------------------------------
    # operator contract: eager runs the CompositeExplicitAutograd reference impl
    # (per-segment copy + cat), not the fused kernel, which only exists after
    # lowering. These pin semantics and arg validation; perf in bench_multi_slice_concat.py.
    # ------------------------------------------------------------------
    @parametrize('rows', [1, 32, 200])
    @parametrize('dtype', ['float16'])
    def test_op_matches_cat_of_slices(self, rows, dtype):
        wide = self._wide(rows, dtype)
        widths = (128,) * len(_W128_OFFSETS)
        n = len(widths)
        out = _MSC([wide], [], [0] * n, list(_W128_OFFSETS), list(widths), [-1] * n)
        self._assert_bitwise_equal(_col_concat_ref(wide, _W128_OFFSETS, widths), out,
                                   "op reference impl")
        self.assertEqual(out.shape, (rows, sum(widths)))

    @parametrize('rows', [200])
    @parametrize('dtype', ['float16'])
    def test_op_handles_mixed_widths(self, rows, dtype):
        wide = self._wide(rows, dtype)
        n = len(_MIXED_WIDTHS)
        out = _MSC([wide], [], [0] * n, list(_MIXED_OFFSETS), list(_MIXED_WIDTHS),
                   [-1] * n)
        self._assert_bitwise_equal(
            _col_concat_ref(wide, _MIXED_OFFSETS, _MIXED_WIDTHS), out, "45 segments mixed widths"
        )

    @parametrize('rows', [200])
    @parametrize('dtype', ['float16'])
    def test_op_handles_non_pow2_width(self, rows, dtype):
        """Model widths are all powers of two, but the template padding branch must work."""
        wide = self._wide(rows, dtype)
        offsets, widths = (100, 500, 900), (24, 48, 12)
        out = _MSC([wide], [], [0] * 3, list(offsets), list(widths), [-1] * 3)
        self._assert_bitwise_equal(_col_concat_ref(wide, offsets, widths), out,
                                   "non power-of-2 widths")

    @parametrize('rows', [32, 200])
    @parametrize('dtype', ['float16'])
    def test_op_masked_segment(self, rows, dtype):
        """A masked segment is equivalent to where(row mask, 0, slice)."""
        wide = self._wide(rows, dtype)
        mask = self._mask(rows)
        offsets, widths = (100, 500, 900), (16, 32, 64)
        out = _MSC([wide], [mask], [0] * 3, list(offsets), list(widths), [0, -1, 0])

        expected = torch.cat([
            _masked_slice_ref(wide, mask, 100, 16),
            wide[..., 500:532],
            _masked_slice_ref(wide, mask, 900, 964 - 900),
        ], dim=-1)
        self._assert_bitwise_equal(expected, out, "masked and bare segments mixed")

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_op_multi_source(self, rows, dtype):
        """src_idx picks which source tensor each segment reads from."""
        a = self._wide(rows, dtype, cols=1024)
        b = self._wide(rows, dtype, cols=512)
        out = _MSC([a, b], [], [0, 1, 0], [100, 64, 700], [32, 16, 48], [-1] * 3)

        expected = torch.cat([a[..., 100:132], b[..., 64:80], a[..., 700:748]], dim=-1)
        self._assert_bitwise_equal(expected, out, "two-source segment plan")

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_op_rejects_out_of_range_segment(self, rows, dtype):
        wide = self._wide(rows, dtype)
        with self.assertRaises(RuntimeError):
            _MSC([wide], [], [0], [_WIDE_COLS - 8], [16], [-1])

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_op_rejects_bad_plan(self, rows, dtype):
        """Mismatched plan lengths or out-of-range indices must raise, not miscompute."""
        wide = self._wide(rows, dtype)
        with self.assertRaises(RuntimeError):
            _MSC([wide], [], [0, 0], [100], [16], [-1])
        with self.assertRaises(RuntimeError):
            _MSC([wide], [], [3], [100], [16], [-1])
        with self.assertRaises(RuntimeError):
            _MSC([wide], [], [0], [100], [16], [0])

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_op_rejects_non_row_mask(self, rows, dtype):
        """A non-[rows, 1] mask broadcasts differently, so it must raise, not guess."""
        wide = self._wide(rows, dtype)
        bad = torch.randint(0, 2, (rows, 16), device=torch.device("npu"),
                            dtype=torch.bool)
        with self.assertRaises(RuntimeError):
            _MSC([wide], [bad], [0], [100], [16], [0])

    # ------------------------------------------------------------------
    # rewrite under static and dynamic shapes
    # ------------------------------------------------------------------
    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    @parametrize('dynamic', [False, True])
    def test_whole_cat_collapses(self, rows, dtype, dynamic):
        wide = self._wide(rows, dtype)
        widths = (128,) * len(_W128_OFFSETS)

        def fn(x):
            return _col_concat(x, _W128_OFFSETS, widths)

        gm = self._run_pass(fn, wide, dynamic=dynamic)

        self.assertEqual(_count(gm.graph, _CAT), 0, "slices should be replaced by the op, cat gone")
        self.assertEqual(len(_op_nodes(gm.graph)), 1)
        self.assertEqual(_count(gm.graph, _SLICE), 0, "slices should be eliminated too")
        self._assert_bitwise_equal(_col_concat_ref(wide, _W128_OFFSETS, widths),
                                   gm(wide), f"15 segments width=128 (dynamic={dynamic})")

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_mixed_widths_from_real_model(self, rows, dtype):
        """L25021 prototype: 45 segments with widths mixed 16/32/128."""
        wide = self._wide(rows, dtype)

        def fn(x):
            return _col_concat(x, _MIXED_OFFSETS, _MIXED_WIDTHS)

        gm = self._run_pass(fn, wide)
        ops = _op_nodes(gm.graph)

        self.assertEqual(_count(gm.graph, _CAT), 0)
        self.assertEqual(len(ops), 1, "45 segments of one base should collapse into one op")
        self.assertEqual(len(ops[0].args[_A_OFFSETS]), 45, "all segments must be preserved")
        self.assertEqual(list(ops[0].args[_A_OFFSETS]), list(_MIXED_OFFSETS),
                         "offset order must not change")
        self._assert_bitwise_equal(
            _col_concat_ref(wide, _MIXED_OFFSETS, _MIXED_WIDTHS), gm(wide), "45 segments mixed widths"
        )

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_narrow_segments(self, rows, dtype):
        """L27947 prototype: 28 segments of width=8, the narrowest case."""
        wide = self._wide(rows, dtype)
        widths = (8,) * len(_W8_OFFSETS)

        def fn(x):
            return _col_concat(x, _W8_OFFSETS, widths)

        gm = self._run_pass(fn, wide)
        self.assertEqual(len(_op_nodes(gm.graph)), 1)
        self._assert_bitwise_equal(_col_concat_ref(wide, _W8_OFFSETS, widths),
                                   gm(wide), "28 segments width=8")

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_dynamic_batch_reruns_with_other_shape(self, rows, dtype):
        wide = self._wide(rows, dtype)
        widths = (16,) * len(_W16_OFFSETS)

        def fn(x):
            return _col_concat(x, _W16_OFFSETS, widths)

        gm = self._run_pass(fn, wide, dynamic=True)
        self.assertEqual(len(_op_nodes(gm.graph)), 1)

        # batch is never 1: dims 0/1 are hard-specialized, so symbols range over [2, inf).
        for other_rows in (2, rows * 3):
            other = self._wide(other_rows, dtype)
            self._assert_bitwise_equal(_col_concat_ref(other, _W16_OFFSETS, widths),
                                       gm(other), f"dynamic batch rows={other_rows}")

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_new_node_carries_fake_meta(self, rows, dtype):
        wide = self._wide(rows, dtype)
        widths = (16,) * len(_W16_OFFSETS)

        def fn(x):
            return _col_concat(x, _W16_OFFSETS, widths)

        node = _op_nodes(self._run_pass(fn, wide).graph)[0]

        self.assertIn('val', node.meta, "new node must carry meta['val'] or lowering has no shape")
        self.assertEqual(node.meta['val'].dtype, torch.float16)
        self.assertEqual(int(node.meta['val'].shape[-1]), sum(widths))

    # ------------------------------------------------------------------
    # masked segments: more common than bare slices in production, and what breaks up runs
    # ------------------------------------------------------------------
    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    @parametrize('dynamic', [False, True])
    def test_masked_slices_collapse(self, rows, dtype, dynamic):
        wide = self._wide(rows, dtype)
        mask = self._mask(rows)
        offsets, widths = (100, 500, 900, 1300), (16, 32, 64, 16)

        def fn(x, m):
            return _CAT([_masked_slice(x, m, o, w)
                         for o, w in zip(offsets, widths)], -1)

        gm = self._run_pass(fn, wide, mask, dynamic=dynamic)
        ops = _op_nodes(gm.graph)

        self.assertEqual(len(ops), 1, "the whole masked slice run should collapse into one op")
        self.assertEqual(_count(gm.graph, _CAT), 0)
        self.assertEqual(_count(gm.graph, _WHERE), 0, "where should be eliminated too")
        self.assertEqual(len(ops[0].args[_A_MASKS]), 1, "one mask must be registered only once")
        self.assertEqual(list(ops[0].args[_A_MASK_IDX]), [0] * 4)

        expected = torch.cat([_masked_slice_ref(wide, mask, o, w)
                              for o, w in zip(offsets, widths)], dim=-1)
        self._assert_bitwise_equal(expected, gm(wide, mask),
                                   f"masked slices (dynamic={dynamic})")

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_interleaved_masked_and_bare_collapse(self, rows, dtype):
        """cat_124 prototype: masked and bare slices strictly alternating.

        Bare-slice-only matching merges nothing here, since no two bare slices are
        adjacent. Covering masked segments joins the run and is the main gain.
        """
        wide = self._wide(rows, dtype)
        mask = self._mask(rows)

        def fn(x, m):
            parts = []
            for i, (off, w) in enumerate(zip(_ALT_OFFSETS, _ALT_WIDTHS)):
                parts.append(_masked_slice(x, m, off, w) if i % 2 == 0
                             else _SLICE(x, -1, off, off + w))
            return _CAT(parts, -1)

        gm = self._run_pass(fn, wide, mask)
        ops = _op_nodes(gm.graph)

        self.assertEqual(len(ops), 1, "an alternating layout should form one run")
        self.assertEqual(len(ops[0].args[_A_OFFSETS]), len(_ALT_OFFSETS))
        self.assertEqual(list(ops[0].args[_A_MASK_IDX]), [0, -1, 0, -1, 0, -1, 0, -1],
                         "masked and bare segments must each be recorded correctly")

        expected = torch.cat([
            _masked_slice_ref(wide, mask, off, w) if i % 2 == 0
            else wide[..., off:off + w]
            for i, (off, w) in enumerate(zip(_ALT_OFFSETS, _ALT_WIDTHS))
        ], dim=-1)
        self._assert_bitwise_equal(expected, gm(wide, mask), "masked and bare slices alternating")

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_mask_reshaped_before_where_is_matched(self, rows, dtype):
        """Broadcasts of the condition (reshape/expand to width) must be stripped."""
        wide = self._wide(rows, dtype)
        mask = self._mask(rows)

        def fn(x, m):
            parts = []
            for off, w in ((100, 16), (500, 32)):
                cond = torch.ops.aten.expand.default(
                    torch.ops.aten.reshape.default(m, [-1, 1]), [x.shape[0], w]
                )
                zero = _FULL([x.shape[0], w], 0, dtype=x.dtype, device=x.device)
                parts.append(_WHERE(cond, zero, _SLICE(x, -1, off, off + w)))
            return _CAT(parts, -1)

        gm = self._run_pass(fn, wide, mask)
        ops = _op_nodes(gm.graph)

        self.assertEqual(len(ops), 1, "a broadcast condition must still count as a row mask")
        self.assertEqual(ops[0].args[_A_MASKS][0].op, "placeholder",
                         "the registered mask must be the one before broadcast")
        expected = torch.cat([_masked_slice_ref(wide, mask, 100, 16),
                              _masked_slice_ref(wide, mask, 500, 32)], dim=-1)
        self._assert_bitwise_equal(expected, gm(wide, mask), "broadcast condition")

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_mask_chain_stops_at_deepest_row_mask(self, rows, dtype):
        """Broadcast stripping must stop at the deepest [rows, 1].

        The producer here is [1, 1], so stripping all the way down lands on a node
        whose row count does not match and drops the segment; the intermediate
        [rows, 1] is the real row mask.
        """
        wide = self._wide(rows, dtype)
        seed = self._mask(1)

        def fn(x, s):
            row = torch.ops.aten.expand.default(s, [x.shape[0], 1])
            parts = []
            for off, w in ((100, 16), (500, 32)):
                cond = torch.ops.aten.expand.default(row, [x.shape[0], w])
                zero = _FULL([x.shape[0], w], 0, dtype=x.dtype, device=x.device)
                parts.append(_WHERE(cond, zero, _SLICE(x, -1, off, off + w)))
            return _CAT(parts, -1)

        gm = self._run_pass(fn, wide, seed)
        ops = _op_nodes(gm.graph)

        self.assertEqual(len(ops), 1, "a qualifying row mask mid-chain must not drop the segment")
        self.assertEqual(len(ops[0].args[_A_MASKS]), 1, "one mask must be registered only once")
        recorded = ops[0].args[_A_MASKS][0]
        self.assertEqual(tuple(recorded.meta["val"].shape), (rows, 1),
                         "the registered mask must be [rows, 1]")
        self.assertNotEqual(recorded.op, "placeholder",
                            "a [1, 1] graph input is not a row mask and must not be registered")

        row = seed.expand(rows, 1)
        expected = torch.cat([_masked_slice_ref(wide, row, 100, 16),
                              _masked_slice_ref(wide, row, 500, 32)], dim=-1)
        self._assert_bitwise_equal(expected, gm(wide, seed), "scalar broadcast mask")

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_two_masks_registered_separately(self, rows, dtype):
        wide = self._wide(rows, dtype)
        m0, m1 = self._mask(rows), self._mask(rows)

        def fn(x, a, b):
            return _CAT([_masked_slice(x, a, 100, 16), _masked_slice(x, b, 500, 32),
                         _masked_slice(x, a, 900, 16)], -1)

        gm = self._run_pass(fn, wide, m0, m1)
        ops = _op_nodes(gm.graph)

        self.assertEqual(len(ops), 1)
        self.assertEqual(len(ops[0].args[_A_MASKS]), 2, "two distinct masks must each be registered")
        self.assertEqual(list(ops[0].args[_A_MASK_IDX]), [0, 1, 0], "mask indices must be reused")
        expected = torch.cat([_masked_slice_ref(wide, m0, 100, 16),
                              _masked_slice_ref(wide, m1, 500, 32),
                              _masked_slice_ref(wide, m0, 900, 16)], dim=-1)
        self._assert_bitwise_equal(expected, gm(wide, m0, m1), "two masks")

    # ------------------------------------------------------------------
    # multi-source: one concat often mixes several model inputs
    # ------------------------------------------------------------------
    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_two_bases_merge_into_one_run(self, rows, dtype):
        """Slices of different bases can share one op; each source costs one pointer arg."""
        a, b = self._wide(rows, dtype), self._wide(rows, dtype)

        def fn(x, y):
            return _CAT([
                _SLICE(x, -1, 100, 116), _SLICE(x, -1, 900, 916),
                _SLICE(y, -1, 500, 532), _SLICE(y, -1, 700, 732),
            ], -1)

        gm = self._run_pass(fn, a, b)
        ops = _op_nodes(gm.graph)

        self.assertEqual(len(ops), 1, "a contiguous multi-source run should collapse into one op")
        self.assertEqual(len(ops[0].args[_A_SRCS]), 2, "both sources must be registered")
        self.assertEqual(list(ops[0].args[_A_SRC_IDX]), [0, 0, 1, 1], "source indices must match")
        expected = torch.cat([a[..., 100:116], a[..., 900:916],
                              b[..., 500:532], b[..., 700:732]], dim=-1)
        self._assert_bitwise_equal(expected, gm(a, b), "two base tensors")

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_source_cap_splits_run(self, rows, dtype):
        a, b = self._wide(rows, dtype), self._wide(rows, dtype)

        def fn(x, y):
            return _CAT([
                _SLICE(x, -1, 100, 116), _SLICE(x, -1, 900, 916),
                _SLICE(y, -1, 500, 532), _SLICE(y, -1, 700, 732),
            ], -1)

        gm = self._run_pass(fn, a, b, max_sources=1)
        ops = _op_nodes(gm.graph)

        self.assertEqual(len(ops), 2, "with a source cap of 1 the run must split per base")
        for node in ops:
            self.assertEqual(len(node.args[_A_SRCS]), 1)
        expected = torch.cat([a[..., 100:116], a[..., 900:916],
                              b[..., 500:532], b[..., 700:732]], dim=-1)
        self._assert_bitwise_equal(expected, gm(a, b), "source cap split")

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_mask_cap_splits_run(self, rows, dtype):
        wide = self._wide(rows, dtype)
        m0, m1 = self._mask(rows), self._mask(rows)

        def fn(x, a, b):
            return _CAT([_masked_slice(x, a, 100, 16), _masked_slice(x, a, 300, 16),
                         _masked_slice(x, b, 500, 32), _masked_slice(x, b, 700, 32)], -1)

        gm = self._run_pass(fn, wide, m0, m1, max_masks=1)
        ops = _op_nodes(gm.graph)

        self.assertEqual(len(ops), 2, "with a mask cap of 1 the run must split per mask")
        for node in ops:
            self.assertEqual(len(node.args[_A_MASKS]), 1)
        expected = torch.cat([_masked_slice_ref(wide, m0, 100, 16),
                              _masked_slice_ref(wide, m0, 300, 16),
                              _masked_slice_ref(wide, m1, 500, 32),
                              _masked_slice_ref(wide, m1, 700, 32)], dim=-1)
        self._assert_bitwise_equal(expected, gm(wide, m0, m1), "mask cap split")

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_masks_disabled_falls_back_to_bare_slices(self, rows, dtype):
        wide = self._wide(rows, dtype)
        mask = self._mask(rows)

        def fn(x, m):
            return _CAT([_masked_slice(x, m, 100, 16), _masked_slice(x, m, 500, 32)], -1)

        gm = self._run_pass(fn, wide, mask, max_masks=0)
        self.assertEqual(len(_op_nodes(gm.graph)), 0, "no rewrite when masks are disabled")

    # ------------------------------------------------------------------
    # slices mixed with other inputs: only contiguous runs merge, order is preserved
    # ------------------------------------------------------------------
    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_slices_mixed_with_dense_input(self, rows, dtype):
        wide = self._wide(rows, dtype)
        dense = torch.randn((rows, 64), dtype=torch.float16, device=torch.device("npu"))
        widths = (16,) * len(_W16_OFFSETS)

        def fn(x, d):
            parts = [_SLICE(x, -1, off, off + w) for off, w in zip(_W16_OFFSETS, widths)]
            return _CAT(parts + [torch.ops.aten.relu.default(d)], -1)

        gm = self._run_pass(fn, wide, dense)

        self.assertEqual(len(_op_nodes(gm.graph)), 1, "the slice run should collapse into one op")
        self.assertEqual(_count(gm.graph, _CAT), 1, "outer cat must stay to join the dense input")
        expected = torch.cat(
            [_col_concat_ref(wide, _W16_OFFSETS, widths), torch.relu(dense)], dim=-1
        )
        self._assert_bitwise_equal(expected, gm(wide, dense), "slice run + dense input")

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_interleaved_slices_split_into_runs(self, rows, dtype):
        wide = self._wide(rows, dtype)
        dense = torch.randn((rows, 8), dtype=torch.float16, device=torch.device("npu"))

        def fn(x, d):
            return _CAT([
                _SLICE(x, -1, 100, 116),
                _SLICE(x, -1, 500, 516),
                torch.ops.aten.relu.default(d),
                _SLICE(x, -1, 900, 916),
                _SLICE(x, -1, 1300, 1316),
            ], -1)

        gm = self._run_pass(fn, wide, dense)

        self.assertEqual(len(_op_nodes(gm.graph)), 2, "each slice run collapses into its own op")
        expected = torch.cat([
            wide[..., 100:116], wide[..., 500:516], torch.relu(dense),
            wide[..., 900:916], wide[..., 1300:1316],
        ], dim=-1)
        self._assert_bitwise_equal(expected, gm(wide, dense), "slices split apart")

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_segment_cap_splits_run(self, rows, dtype):
        wide = self._wide(rows, dtype)
        widths = (128,) * len(_W128_OFFSETS)

        def fn(x):
            return _col_concat(x, _W128_OFFSETS, widths)

        gm = self._run_pass(fn, wide, max_segments=4)
        ops = _op_nodes(gm.graph)

        self.assertEqual(len(ops), 4, "15 segments capped at 4 per group make 4 groups")
        for node in ops:
            self.assertLessEqual(len(node.args[_A_OFFSETS]), 4, "group exceeds the segment cap")
        self._assert_bitwise_equal(_col_concat_ref(wide, _W128_OFFSETS, widths),
                                   gm(wide), "segment cap split")

    # ------------------------------------------------------------------
    # profitability: merging costs one dispatch, so a segment paying less than two loses
    # ------------------------------------------------------------------
    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_whole_inputs_alone_not_rewritten(self, rows, dtype):
        """aclnnCat reads contiguous inputs directly; folding them in adds a dispatch."""
        a = self._wide(rows, dtype, cols=64)
        b = self._wide(rows, dtype, cols=32)

        def fn(x, y):
            return _CAT([x, y], -1)

        self.assertEqual(len(_op_nodes(self._run_pass(fn, a, b).graph)), 0,
                         "a plain concat of two dense inputs must not be rewritten")

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_whole_input_joins_paying_run(self, rows, dtype):
        """A whole input saves nothing alone, but rides along a run that already pays."""
        wide = self._wide(rows, dtype)
        dense = self._wide(rows, dtype, cols=64)

        def fn(x, d):
            return _CAT([_SLICE(x, -1, 100, 116), d, _SLICE(x, -1, 900, 916)], -1)

        gm = self._run_pass(fn, wide, dense)
        ops = _op_nodes(gm.graph)

        self.assertEqual(len(ops), 1, "two slices already pay off, so the whole input can join")
        self.assertEqual(_count(gm.graph, _CAT), 0)
        expected = torch.cat([wide[..., 100:116], dense, wide[..., 900:916]], dim=-1)
        self._assert_bitwise_equal(expected, gm(wide, dense), "whole input between slices")

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_single_masked_slice_with_whole_input_not_rewritten(self, rows, dtype):
        """Only one segment costs a dispatch, so the rewrite gains nothing."""
        wide = self._wide(rows, dtype)
        mask = self._mask(rows)
        dense = self._wide(rows, dtype, cols=64)

        def fn(x, m, d):
            return _CAT([_masked_slice(x, m, 100, 16), d], -1)

        self.assertEqual(len(_op_nodes(self._run_pass(fn, wide, mask, dense).graph)), 0)

    # ------------------------------------------------------------------
    # must reject: a false match silently miscomputes, far worse than not optimizing
    # ------------------------------------------------------------------
    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_single_slice_not_rewritten(self, rows, dtype):
        """One slice means one dispatch before and after, so there is no gain."""
        wide = self._wide(rows, dtype)
        dense = torch.randn((rows, 16), dtype=torch.float16, device=torch.device("npu"))

        def fn(x, d):
            return _CAT([_SLICE(x, -1, 100, 116), torch.ops.aten.relu.default(d)], -1)

        gm = self._run_pass(fn, wide, dense)
        self.assertEqual(len(_op_nodes(gm.graph)), 0)

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_row_direction_cat_not_rewritten(self, rows, dtype):
        wide = self._wide(rows, dtype)

        def fn(x):
            return _CAT([_SLICE(x, -1, 0, 16), _SLICE(x, -1, 64, 80)], 0)

        self.assertEqual(len(_op_nodes(self._run_pass(fn, wide).graph)), 0)

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_slice_on_other_dim_not_rewritten(self, rows, dtype):
        """The slice axis differs from the concat axis, so these are not column segments."""
        wide = self._wide(rows, dtype)

        def fn(x):
            return _CAT([_SLICE(x, 0, 0, 8), _SLICE(x, 0, 8, 16)], -1)

        self.assertEqual(len(_op_nodes(self._run_pass(fn, wide).graph)), 0)

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_strided_slice_not_rewritten(self, rows, dtype):
        wide = self._wide(rows, dtype)

        def fn(x):
            return _CAT([_SLICE(x, -1, 0, 32, 2), _SLICE(x, -1, 64, 96, 2)], -1)

        self.assertEqual(len(_op_nodes(self._run_pass(fn, wide).graph)), 0,
                         "strided slices are not fixed contiguous segments, must be rejected")

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_mixed_dtype_not_rewritten(self, rows, dtype):
        """cat promotes mismatched dtypes, the op does not, so leave it to the original."""
        a, b = self._wide(rows, 'float16'), self._wide(rows, 'float32')

        def fn(x, y):
            return _CAT([_SLICE(x, -1, 0, 16), _SLICE(y, -1, 0, 16)], -1)

        self.assertEqual(len(_op_nodes(self._run_pass(fn, a, b).graph)), 0)

    @parametrize('rows', [8])
    @parametrize('dtype', ['float16'])
    def test_three_dim_not_rewritten(self, rows, dtype):
        """The op only supports 2D; higher rank goes back to the original cat."""
        x = torch.randn((rows, 4, 256), dtype=torch.float16, device=torch.device("npu"))

        def fn(t):
            return _CAT([_SLICE(t, -1, 0, 16), _SLICE(t, -1, 64, 80)], -1)

        self.assertEqual(len(_op_nodes(self._run_pass(fn, x).graph)), 0)

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_computed_base_not_rewritten(self, rows, dtype):
        """Skip computed bases so they are not forced to materialize early.

        In the target case the wide table is a model input and already landed. For a
        computed tensor, realize_input in lowering would demand a contiguous buffer
        first, which can block fusion on the producer side.
        """
        wide = self._wide(rows, dtype)

        def fn(x):
            base = torch.ops.aten.mul.Tensor(x, 2.0)
            return _CAT([_SLICE(base, -1, 0, 16), _SLICE(base, -1, 64, 80)], -1)

        self.assertEqual(len(_op_nodes(self._run_pass(fn, wide).graph)), 0,
                         "a base that is not a graph input must fall back to the original cat")

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_masked_computed_base_not_rewritten(self, rows, dtype):
        """The selected value of a masked segment must also come from a graph input."""
        wide = self._wide(rows, dtype)
        mask = self._mask(rows)

        def fn(x, m):
            base = torch.ops.aten.mul.Tensor(x, 2.0)
            return _CAT([_masked_slice(base, m, 100, 16),
                         _masked_slice(base, m, 500, 32)], -1)

        self.assertEqual(len(_op_nodes(self._run_pass(fn, wide, mask).graph)), 0)

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_elementwise_mask_not_rewritten(self, rows, dtype):
        """An elementwise mask is not row-wise, so the op cannot express it."""
        wide = self._wide(rows, dtype)

        def fn(x, m0, m1):
            zero0 = _FULL([x.shape[0], 16], 0, dtype=x.dtype, device=x.device)
            zero1 = _FULL([x.shape[0], 16], 0, dtype=x.dtype, device=x.device)
            return _CAT([_WHERE(m0, zero0, _SLICE(x, -1, 100, 116)),
                         _WHERE(m1, zero1, _SLICE(x, -1, 500, 516))], -1)

        wide_mask0 = torch.randint(0, 2, (rows, 16), device=torch.device("npu"),
                                   dtype=torch.bool)
        wide_mask1 = torch.randint(0, 2, (rows, 16), device=torch.device("npu"),
                                   dtype=torch.bool)
        gm = self._run_pass(fn, wide, wide_mask0, wide_mask1)
        self.assertEqual(len(_op_nodes(gm.graph)), 0, "an elementwise mask must be rejected")

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_nonzero_fill_not_rewritten(self, rows, dtype):
        wide = self._wide(rows, dtype)
        mask = self._mask(rows)

        def fn(x, m):
            parts = []
            for off, w in ((100, 16), (500, 32)):
                one = _FULL([x.shape[0], w], 1, dtype=x.dtype, device=x.device)
                parts.append(_WHERE(m, one, _SLICE(x, -1, off, off + w)))
            return _CAT(parts, -1)

        self.assertEqual(len(_op_nodes(self._run_pass(fn, wide, mask).graph)), 0)

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_where_between_two_slices_not_rewritten(self, rows, dtype):
        """Selecting between two slices reads two places, not zero-fill; unsupported."""
        wide = self._wide(rows, dtype)
        mask = self._mask(rows)

        def fn(x, m):
            return _CAT([
                _WHERE(m, _SLICE(x, -1, 100, 116), _SLICE(x, -1, 200, 216)),
                _WHERE(m, _SLICE(x, -1, 500, 516), _SLICE(x, -1, 600, 616)),
            ], -1)

        self.assertEqual(len(_op_nodes(self._run_pass(fn, wide, mask).graph)), 0)

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_slice_shared_with_other_consumer_is_kept(self, rows, dtype):
        wide = self._wide(rows, dtype)

        def fn(x):
            a = _SLICE(x, -1, 55818, 55834)
            b = _SLICE(x, -1, 56436, 56452)
            # a feeds both the cat and a pointwise op, the form Inductor fuses.
            return _CAT([a, b], -1), torch.ops.aten.mul.Tensor(a, 2.0)

        gm = self._run_pass(fn, wide)
        self.assertEqual(len(_op_nodes(gm.graph)), 1, "the cat path should still be rewritten")
        self.assertGreaterEqual(_count(gm.graph, _SLICE), 1,
                                "a slice used elsewhere must not be deleted")
        expected = fn(wide)
        actual = gm(wide)
        self._assert_bitwise_equal(expected[0], actual[0], "cat result with a shared slice")
        self._assert_bitwise_equal(expected[1], actual[1], "other consumer of a shared slice")

    # ------------------------------------------------------------------
    # end to end; needs the flag on, otherwise pass and lowering are not registered
    # ------------------------------------------------------------------
    @unittest.skipUnless(_ENABLED, _NEEDS_FLAG)
    @parametrize('rows', [200])
    @parametrize('dtype', ['float16'])
    @parametrize('dynamic', [False, True])
    def test_compile_end_to_end(self, rows, dtype, dynamic):
        """The dynamic case marks only batch as symbolic and cannot use
        ``torch.compile(dynamic=True)``: that also symbolizes the column count, and
        the pass needs a static last dim to read constant offsets, so it silently
        falls back to aten.cat. Values stay correct but the pass is not covered.
        """
        widths = (16,) * len(_W16_OFFSETS)
        wide = self._wide(rows, dtype)

        def fn(x):
            return _col_concat_ref(x, _W16_OFFSETS, widths)

        with torch.no_grad(), _pass_spy() as stats:
            compiled = torch.compile(fn, backend="inductor", dynamic=False)
            if dynamic:
                torch._dynamo.mark_dynamic(wide, 0)
            self._assert_bitwise_equal(fn(wide), compiled(wide), "compiled artifact")
            self.assertGreater(stats["nodes"], 0,
                               f"pass did not fire (dynamic={dynamic}), still aten.cat")
            if dynamic:
                # another batch must reuse the same artifact. never 1: dims 0/1 are
                # hard-specialized, backed symbols are [2, inf), so 1 recompiles.
                other = self._wide(rows * 2, dtype)
                torch._dynamo.mark_dynamic(other, 0)
                self._assert_bitwise_equal(fn(other), compiled(other), "dynamic batch artifact")

    @unittest.skipUnless(_ENABLED, _NEEDS_FLAG)
    @parametrize('rows', [200])
    @parametrize('dtype', ['float16'])
    def test_compile_masked_end_to_end(self, rows, dtype):
        wide = self._wide(rows, dtype)
        mask = self._mask(rows)
        plan = ((100, 16), (500, 32), (900, 64), (1300, 16))

        def fn(x, m):
            return torch.cat([_masked_slice_ref(x, m, off, w) for off, w in plan], dim=-1)

        with torch.no_grad(), _pass_spy() as stats:
            compiled = torch.compile(fn, backend="inductor", dynamic=False)
            self._assert_bitwise_equal(fn(wide, mask), compiled(wide, mask), "masked artifact")
            self.assertGreater(stats["nodes"], 0, "masked segments were not rewritten")

    @unittest.skipUnless(_ENABLED, _NEEDS_FLAG)
    @parametrize('rows', [200])
    @parametrize('dtype', ['float16'])
    def test_compile_aliased_masks_end_to_end(self, rows, dtype):
        """Two mask nodes landing on the same buffer must still compile.

        One row mask is squeezed and reshaped twice; squeeze is not a stripped
        broadcast, so both chains keep a reshape node and the pass registers two
        masks. In lowering both views point at the same ComputedBuffer, and
        def_kernel registers args by buffer name, so the entries overwrite each
        other, the first arg is left undefined, and the rendered kernel raises
        NameError. is_live_type in the production model has this exact shape.
        """
        wide = self._wide(rows, dtype)
        flag = self._mask(rows)
        plan = ((100, 16), (500, 32), (900, 64), (1300, 16))

        def fn(x, f):
            cond = torch.logical_not(f)
            m0 = cond.squeeze().reshape(-1, 1)
            m1 = cond.squeeze().reshape(-1, 1)
            picks = (m0, m1, m1, m0)
            return torch.cat(
                [_masked_slice_ref(x, m, off, w)
                 for m, (off, w) in zip(picks, plan)], dim=-1)

        with torch.no_grad(), _pass_spy() as stats:
            compiled = torch.compile(fn, backend="inductor", dynamic=False)
            self._assert_bitwise_equal(fn(wide, flag), compiled(wide, flag),
                                       "aliased masks artifact")
            self.assertGreater(stats["nodes"], 0, "aliased mask segments were not rewritten")

    @unittest.skipUnless(_ENABLED, _NEEDS_FLAG)
    @parametrize('rows', [2, 200])
    @parametrize('dtype', ['float16'])
    def test_compile_epilogue_is_not_fused_away(self, rows, dtype):
        """A pointwise consumer of the concat must not be fused into the template.

        The template stores every segment with its own tl.store and renders no
        store_output hook, so an epilogue fused into it has nowhere to be codegened and
        is skipped: the kernel returns the bare concat and the consumer disappears. In
        the model this was a clamp between the concat and an mm, which left the mm reading
        unclamped values. The lowering marks the output no-fuse, so the clamp stays a
        kernel of its own while the rewrite still happens.
        """
        widths = (16,) * len(_W16_OFFSETS)
        wide = self._wide(rows, dtype)
        # exactly representable in fp16, so it survives into the kernel source as written
        bound = 0.375

        def fn(x):
            return torch.clamp(_col_concat_ref(x, _W16_OFFSETS, widths), -bound, bound)

        with torch.no_grad(), _pass_spy() as stats:
            compiled = torch.compile(fn, backend="inductor", dynamic=False)
            actual, codes = run_and_get_code(compiled, wide)
            self.assertGreater(stats["nodes"], 0, "pass did not fire, still aten.cat")
            self._assert_bitwise_equal(fn(wide), actual, "clamp after the concat")

            # arg_SRC0 is the first source pointer the template renders
            source = _kernel_source(codes[0], "arg_SRC0")
            self.assertIsNotNone(
                source, "no template kernel in the output code, lowering fell back")
            # whichever form the clamp takes, none of it belongs in the template
            for trace in (str(bound), "maximum", "minimum", "clamp("):
                self.assertNotIn(trace, source,
                                 f"clamp was fused into the template ({trace})")


    # ------------------------------------------------------------------
    # input dedup in lowering, pinned here since the case above needs the whole chain
    # ------------------------------------------------------------------
    def test_dedup_inputs_merges_aliased_buffers(self):
        """Duplicate inputs on one buffer merge, indices are remapped, and -1 stays."""
        first, second = _StubInput("buf5"), _StubInput("buf5")
        nodes, indices = _dedup_inputs([first, second], [0, -1, 1, 1, 0])
        self.assertEqual(nodes, [first])
        self.assertEqual(indices, [0, -1, 0, 0, 0])

    def test_dedup_inputs_keeps_distinct_buffers(self):
        first, second = _StubInput("buf5"), _StubInput("buf7")
        nodes, indices = _dedup_inputs([first, second], [1, 0, -1])
        self.assertEqual(nodes, [first, second])
        self.assertEqual(indices, [1, 0, -1])

    def test_dedup_inputs_rejects_same_buffer_different_layout(self):
        """Same name, different layouts fight over one arg slot; caller must fall back."""
        broadcast = _StubInput("buf5", stride=(0, 1))
        contiguous = _StubInput("buf5", stride=(1, 1))
        self.assertEqual(_dedup_inputs([broadcast, contiguous], [0, 1]), (None, None))


instantiate_parametrized_tests(TestMultiSliceConcatPass)


if __name__ == "__main__":
    run_tests()
