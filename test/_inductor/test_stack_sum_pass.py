import torch
import torch.fx as fx
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu  # noqa: F401
from torch_npu._inductor.fx_passes.ascend_custom_passes.ascend_graph_pass import (
    stack_sum_to_add_chain_pass,
)


# From real model output code: a [B, 69876] fp16 wide feature table, cut into fixed-width
# column blocks that are stacked and summed. All offsets are compile-time constants.
_WIDE_COLS = 69876
_GROUP_W128_OFFSETS = (55834, 56452, 45421, 44203, 44744, 45962, 60741, 61328,
                       62267, 65602, 66495, 66956, 67445, 68021, 69060)
_USER_W128_OFFSETS = (55580, 59302)
_GROUP_W16_OFFSETS = (55818, 56436, 45405, 44187, 44728, 45946, 61312, 62251,
                      69044, 68199, 68221, 69822)
# Widest stack pooling in the model (28-way); used to stress accumulation error.
_GROUP_W8_OFFSETS = (48025, 47595, 46868, 47890, 48779, 49346, 47769, 49082,
                     49180, 48337, 48467, 48597, 48402, 48532, 48714, 49240,
                     48070, 48204, 48119, 11076, 14257, 18643, 11922, 15103,
                     19489, 12752, 15933, 20319)
# fp16 relative precision, used as the magnitude of one ulp.
_FP16_EPS = 2 ** -10


def _count(graph, target):
    return len([n for n in graph.nodes if n.op == "call_function" and n.target == target])


def _fm_pool(wide, offsets, width):
    """Reproduce the [aten.slice, aten.stack] -> sum combination from output code."""
    slices = [
        torch.ops.aten.slice.Tensor(wide, 1, off, off + width) for off in offsets
    ]
    cat = torch.ops.aten.cat.default(slices, 0)
    view = torch.ops.aten.reshape.default(cat, [len(offsets), wide.shape[0], width])
    return torch.ops.aten.sum.dim_IntList(view, [0])


def _fm_pool_ref(wide, offsets, width):
    return torch.stack([wide[:, off:off + width] for off in offsets]).sum(0)


class ViewCatSumModel(torch.nn.Module):
    """Stack form after Inductor simplification: cat(dim=0) -> reshape -> sum(dim=0)."""

    def forward(self, t1, t2, t3):
        cat = torch.ops.aten.cat.default([t1, t2, t3], 0)
        view = torch.ops.aten.reshape.default(cat, [3, t1.shape[0], t1.shape[1]])
        return torch.ops.aten.sum.dim_IntList(view, [0])


class UnsqueezeCatSumModel(torch.nn.Module):
    """Original stack form: unsqueeze -> cat -> sum."""

    def forward(self, t1, t2, t3):
        cat = torch.ops.aten.cat.default(
            [
                torch.ops.aten.unsqueeze.default(t1, 0),
                torch.ops.aten.unsqueeze.default(t2, 0),
                torch.ops.aten.unsqueeze.default(t3, 0),
            ],
            0,
        )
        return torch.ops.aten.sum.dim_IntList(cat, [0])


class TestStackSumPass(TestUtils):
    def _assert_close(self, expected, actual, dtype):
        # At magnitude ~6000 one fp16 ulp is already several units, so compare relative error.
        tol = 1e-2 if dtype in ('float16', 'bfloat16') else 1e-4
        self.assertEqual(expected, actual, atol=tol, rtol=tol)

    def _trace(self, model, *tensors):
        gm = fx.symbolic_trace(model)
        ShapeProp(gm).propagate(*tensors)
        return gm

    def _run_pass(self, model, *tensors):
        gm = self._trace(model, *tensors)
        stack_sum_to_add_chain_pass(gm.graph)
        gm.recompile()
        return gm

    @parametrize('shape', [(8, 16)])
    @parametrize('dtype', ['float32', 'float16'])
    def test_view_cat_sum_rewritten(self, shape, dtype):
        tensors = [self._generate_tensor(shape, dtype) for _ in range(3)]
        model = ViewCatSumModel()
        gm = self._run_pass(model, *tensors)

        self.assertEqual(_count(gm.graph, torch.ops.aten.cat.default), 0, "cat should be eliminated")
        self.assertEqual(_count(gm.graph, torch.ops.aten.sum.dim_IntList), 0, "sum should be eliminated")
        self.assertEqual(_count(gm.graph, torch.ops.aten.add.Tensor), 2, "3 inputs should give 2 adds")
        self._assert_close(model(*tensors), gm(*tensors), dtype)

    @parametrize('shape', [(8, 16)])
    @parametrize('dtype', ['float16'])
    def test_low_precision_accumulates_in_fp32(self, shape, dtype):
        tensors = [self._generate_tensor(shape, dtype) for _ in range(3)]
        gm = self._run_pass(ViewCatSumModel(), *tensors)

        casts = [
            n for n in gm.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.prims.convert_element_type.default
        ]
        self.assertEqual(len(casts), 4, "3 upcasts plus 1 downcast back to the original dtype")
        self.assertEqual(gm(*tensors).dtype, torch.float16, "output dtype should be unchanged")

    @parametrize('shape', [(8, 16)])
    @parametrize('dtype', ['float32'])
    def test_unsqueeze_cat_sum_rewritten(self, shape, dtype):
        tensors = [self._generate_tensor(shape, dtype) for _ in range(3)]
        model = UnsqueezeCatSumModel()
        gm = self._run_pass(model, *tensors)

        self.assertEqual(_count(gm.graph, torch.ops.aten.cat.default), 0, "cat should be eliminated")
        self.assertEqual(_count(gm.graph, torch.ops.aten.add.Tensor), 2)
        self._assert_close(model(*tensors), gm(*tensors), dtype)

    @parametrize('shape', [(8, 16)])
    @parametrize('dtype', ['float32'])
    def test_keepdim_preserves_shape(self, shape, dtype):
        class M(torch.nn.Module):
            def forward(self, t1, t2):
                cat = torch.ops.aten.cat.default([t1, t2], 0)
                view = torch.ops.aten.reshape.default(cat, [2, t1.shape[0], t1.shape[1]])
                return torch.ops.aten.sum.dim_IntList(view, [0], True)

        tensors = [self._generate_tensor(shape, dtype) for _ in range(2)]
        model = M()
        gm = self._run_pass(model, *tensors)

        self.assertEqual(_count(gm.graph, torch.ops.aten.cat.default), 0)
        self._assert_close(model(*tensors), gm(*tensors), dtype)

    @parametrize('shape', [(8, 16)])
    @parametrize('dtype', ['float32'])
    def test_slices_of_one_base(self, shape, dtype):
        """Real model form: column slices of a single wide table, stacked and summed."""
        class M(torch.nn.Module):
            def forward(self, wide):
                slices = [
                    torch.ops.aten.slice.Tensor(wide, 1, off, off + 4)
                    for off in (0, 8, 20, 36)
                ]
                cat = torch.ops.aten.cat.default(slices, 0)
                view = torch.ops.aten.reshape.default(cat, [4, wide.shape[0], 4])
                return torch.ops.aten.sum.dim_IntList(view, [0])

        wide = self._generate_tensor((8, 64), dtype)
        model = M()
        gm = self._run_pass(model, wide)

        self.assertEqual(_count(gm.graph, torch.ops.aten.cat.default), 0, "cat should be eliminated")
        self.assertEqual(_count(gm.graph, torch.ops.aten.slice.Tensor), 4, "slices should survive as views")
        self._assert_close(model(wide), gm(wide), dtype)

    # ------------------------------------------------------------------
    # Real model scenario: symbolic batch, fp16, real offsets, traced by make_fx so nodes carry
    # meta['val']. Covers the propagate_fake_tensor path that ShapeProp never reaches.
    # ------------------------------------------------------------------
    def _wide_tensor(self, rows, dtype, scale=0.1):
        return torch.randn(
            (rows, _WIDE_COLS), dtype=eval('torch.' + dtype), device=torch.device("npu")
        ) * scale

    @staticmethod
    def _fp32_golden(wide, offsets, width):
        return _fm_pool_ref(wide.float(), offsets, width)

    def _assert_not_worse_than(self, golden, baseline, candidate, note):
        """Candidate error against the fp32 golden must not exceed the baseline, plus one fp16 ulp.

        Both paths accumulate in fp32 and round once on write-back, so results are normally
        bitwise-identical; the extra ulp tolerates last-bit differences from accumulation order.
        """
        err_base = (baseline.float() - golden).abs().max().item()
        err_cand = (candidate.float() - golden).abs().max().item()
        one_ulp = golden.abs().max().item() * _FP16_EPS
        self.assertLessEqual(
            err_cand,
            err_base + one_ulp,
            f"{note}: rewritten error {err_cand:.3e} exceeds the original {err_base:.3e} plus one ulp {one_ulp:.3e}",
        )
        self.assertEqual(baseline, candidate, atol=one_ulp, rtol=_FP16_EPS)

    @parametrize('rows', [64])
    @parametrize('dtype', ['float16'])
    def test_accuracy_28way_not_worse_than_unrewritten(self, rows, dtype):
        """Graph-level A/B against the fp32 result: both runs execute the same operators, so any
        error difference can only come from this pass. Comparing against eager cannot isolate it.
        """
        wide = self._wide_tensor(rows, dtype, scale=1.0)

        def fn(x):
            return _fm_pool(x, _GROUP_W8_OFFSETS, 8)

        baseline_gm = make_fx(fn, tracing_mode="symbolic")(wide)
        rewritten_gm = make_fx(fn, tracing_mode="symbolic")(wide)
        stack_sum_to_add_chain_pass(rewritten_gm.graph)
        rewritten_gm.recompile()

        self.assertEqual(_count(baseline_gm.graph, torch.ops.aten.cat.default), 1, "baseline graph should keep the cat")
        self.assertEqual(_count(rewritten_gm.graph, torch.ops.aten.cat.default), 0, "rewritten graph should eliminate the cat")

        golden = self._fp32_golden(wide, _GROUP_W8_OFFSETS, 8)
        self._assert_not_worse_than(golden, baseline_gm(wide), rewritten_gm(wide), "28-way stack pooling")

    def _make_fx_symbolic(self, fn, *tensors):
        gm = make_fx(fn, tracing_mode="symbolic")(*tensors)
        stack_sum_to_add_chain_pass(gm.graph)
        gm.recompile()
        return gm

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_real_model_fm_pooling(self, rows, dtype):
        """Prototype of output code L20616: 15-way width=128 column slice stack sum."""
        wide = self._wide_tensor(rows, dtype)

        def fn(x):
            return _fm_pool(x, _GROUP_W128_OFFSETS, 128)

        gm = self._make_fx_symbolic(fn, wide)

        self.assertEqual(_count(gm.graph, torch.ops.aten.cat.default), 0, "cat should be eliminated")
        self.assertEqual(_count(gm.graph, torch.ops.aten.sum.dim_IntList), 0, "sum should be eliminated")
        self.assertEqual(_count(gm.graph, torch.ops.aten.add.Tensor), 14, "15 inputs should give 14 adds")
        self._assert_close(_fm_pool_ref(wide, _GROUP_W128_OFFSETS, 128), gm(wide), dtype)

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_real_model_mul_of_two_pools(self, rows, dtype):
        """Prototype of triton_per_fused_mul_stack_sum_16: two pooling results multiplied."""
        wide = self._wide_tensor(rows, dtype)

        def fn(x):
            user = _fm_pool(x, _USER_W128_OFFSETS, 128)
            group = _fm_pool(x, _GROUP_W128_OFFSETS, 128)
            return torch.ops.aten.mul.Tensor(user, group)

        gm = self._make_fx_symbolic(fn, wide)

        self.assertEqual(_count(gm.graph, torch.ops.aten.cat.default), 0, "both cats should be eliminated")
        self.assertEqual(_count(gm.graph, torch.ops.aten.add.Tensor), 15, "(2-1) + (15-1)")
        expected = (
            _fm_pool_ref(wide, _USER_W128_OFFSETS, 128)
            * _fm_pool_ref(wide, _GROUP_W128_OFFSETS, 128)
        )
        self._assert_close(expected, gm(wide), dtype)

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_rewritten_nodes_carry_fake_meta(self, rows, dtype):
        """Inserted nodes must carry meta['val'], otherwise Inductor lowering gets no shape."""
        wide = self._wide_tensor(rows, dtype)

        def fn(x):
            return _fm_pool(x, _GROUP_W16_OFFSETS, 16)

        gm = self._make_fx_symbolic(fn, wide)

        inserted = [
            n for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target in (torch.ops.aten.add.Tensor,
                             torch.ops.prims.convert_element_type.default)
        ]
        self.assertTrue(inserted, "add chain and cast nodes should be inserted")
        for node in inserted:
            self.assertIn('val', node.meta, f"{node.name} is missing meta['val']")
        output_node = [n for n in gm.graph.nodes if n.op == "output"][0]
        result = output_node.args[0]
        while isinstance(result, (list, tuple)):
            result = result[0]
        self.assertEqual(result.meta['val'].dtype, torch.float16, "output dtype should stay fp16")

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_symbolic_batch_is_matched(self, rows, dtype):
        """Shape comparison must hold for a symbolic batch, or the pass never fires when dynamic."""
        wide = self._wide_tensor(rows, dtype)

        def fn(x):
            return _fm_pool(x, _GROUP_W16_OFFSETS, 16)

        traced = make_fx(fn, tracing_mode="symbolic")(wide)
        # SymInt is neither hashable nor directly comparable, so only check per-dim types.
        has_symbolic_dim = any(
            isinstance(dim, torch.SymInt)
            for node in traced.graph.nodes
            if 'val' in node.meta and hasattr(node.meta['val'], 'shape')
            for dim in node.meta['val'].shape
        )
        self.assertTrue(
            has_symbolic_dim,
            "trace must carry symbolic shapes, otherwise this case does not exercise dynamic shape",
        )

        stack_sum_to_add_chain_pass(traced.graph)
        traced.recompile()
        self.assertEqual(_count(traced.graph, torch.ops.aten.cat.default), 0)

        other = self._wide_tensor(rows * 2, dtype)
        self._assert_close(_fm_pool_ref(other, _GROUP_W16_OFFSETS, 16), traced(other), dtype)

    @parametrize('shape', [(8, 16)])
    @parametrize('dtype', ['float32'])
    def test_multi_user_cat_not_rewritten(self, shape, dtype):
        class M(torch.nn.Module):
            def forward(self, t1, t2):
                cat = torch.ops.aten.cat.default([t1, t2], 0)
                view = torch.ops.aten.reshape.default(cat, [2, t1.shape[0], t1.shape[1]])
                return torch.ops.aten.sum.dim_IntList(view, [0]), cat

        tensors = [self._generate_tensor(shape, dtype) for _ in range(2)]
        gm = self._run_pass(M(), *tensors)

        self.assertEqual(_count(gm.graph, torch.ops.aten.cat.default), 1, "cat has another user, so it must not be rewritten")
        self.assertEqual(_count(gm.graph, torch.ops.aten.add.Tensor), 0)

    @parametrize('shape', [(8, 16)])
    @parametrize('dtype', ['float32'])
    def test_sum_on_other_dim_not_rewritten(self, shape, dtype):
        class M(torch.nn.Module):
            def forward(self, t1, t2):
                cat = torch.ops.aten.cat.default([t1, t2], 0)
                view = torch.ops.aten.reshape.default(cat, [2, t1.shape[0], t1.shape[1]])
                return torch.ops.aten.sum.dim_IntList(view, [1])

        tensors = [self._generate_tensor(shape, dtype) for _ in range(2)]
        gm = self._run_pass(M(), *tensors)

        self.assertEqual(_count(gm.graph, torch.ops.aten.cat.default), 1, "a sum over a non-stack axis must not be rewritten")

    @parametrize('shape', [(8, 16)])
    @parametrize('dtype', ['int32'])
    def test_integer_dtype_not_rewritten(self, shape, dtype):
        """Integer sum promotes to int64, which an add chain does not, so leave it untouched."""
        tensors = [self._generate_tensor(shape, dtype) for _ in range(3)]
        gm = self._run_pass(ViewCatSumModel(), *tensors)

        self.assertEqual(_count(gm.graph, torch.ops.aten.cat.default), 1)

    @parametrize('shape', [(8, 16)])
    @parametrize('dtype', ['float32'])
    def test_compile_cases(self, shape, dtype):
        def op_calc(t1, t2, t3):
            return torch.stack([t1, t2, t3]).sum(0)

        tensors = [self._generate_tensor(shape, dtype) for _ in range(3)]
        std_result = op_calc(*tensors)
        with torch.no_grad():
            compiled = torch.compile(op_calc, backend="inductor")
            self._assert_close(std_result, compiled(*tensors), dtype)

    @parametrize('rows', [32])
    @parametrize('dtype', ['float16'])
    def test_compile_real_pattern_dynamic_batch(self, rows, dtype):
        def op_calc(x):
            return _fm_pool_ref(x, _GROUP_W16_OFFSETS, 16)

        wide = self._wide_tensor(rows, dtype)
        with torch.no_grad():
            compiled = torch.compile(op_calc, backend="inductor", dynamic=True)
            self._assert_close(op_calc(wide), compiled(wide), dtype)
            other = self._wide_tensor(rows * 2, dtype)
            self._assert_close(op_calc(other), compiled(other), dtype)

    @parametrize('rows', [64])
    @parametrize('dtype', ['float16'])
    def test_compile_accuracy_against_fp32_golden(self, rows, dtype):
        """Eager fp16 error is the yardstick, so the bound holds across data magnitudes."""
        def op_calc(x):
            return _fm_pool_ref(x, _GROUP_W8_OFFSETS, 8)

        wide = self._wide_tensor(rows, dtype, scale=1.0)
        golden = self._fp32_golden(wide, _GROUP_W8_OFFSETS, 8)
        with torch.no_grad():
            compiled = torch.compile(op_calc, backend="inductor")
            self._assert_not_worse_than(golden, op_calc(wide), compiled(wide), "end-to-end 28-way pooling")


instantiate_parametrized_tests(TestStackSumPass)


if __name__ == "__main__":
    run_tests()
