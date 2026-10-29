import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import unittest
from unittest.mock import patch
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize
from torch_npu._inductor.fx_passes.ascend_custom_passes import ascend_graph_pass


@instantiate_parametrized_tests
class TestAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.ns = vars(ascend_graph_pass)

    def check(self, fn, args, indexed, legacy=None, **limits):
        # Structural checks exercise fusion independently of the native Cat budget.
        limits.setdefault("aclnn_copy_free_inputs", 0)
        gm = make_fx(fn, tracing_mode='fake')(*args)
        expected = gm(*args)
        self.ns['multi_slice_concat_pass'](gm.graph, **limits)
        gm.graph.lint()
        gm.recompile()
        torch.testing.assert_close(gm(*args), expected, rtol=0, atol=0, equal_nan=True)
        count = lambda target: sum(n.target is target for n in gm.graph.nodes)
        self.assertEqual(count(self.ns['INDEXED_SLICE_CONCAT_TARGET']), indexed)
        if legacy is not None:
            self.assertEqual(count(self.ns['MULTI_SLICE_CONCAT_TARGET']), legacy)
        before = str(gm.graph)
        self.ns['multi_slice_concat_pass'](gm.graph, **limits)
        self.assertEqual(str(gm.graph), before, 'pass must be idempotent')
        return gm

    def test_645_slices_mixed_sum_and_dtype(self):
        def fn(x, s):
            return torch.cat([torch.ops.prims.convert_element_type.default(
                torch.ops.aten.slice.Tensor(x, 1, i, i+1), torch.float32
            ) for i in range(645)] + [s], dim=1)
        self.check(fn, (torch.randn(3, 700, dtype=torch.bfloat16), torch.randn(3, 1)), 1, 0)

    def test_order_duplicates_negative_bounds_and_noncontiguous(self):
        def fn(x):
            return torch.cat([x[:, -7:-2], x[:, 4:99], x[:, 4:99], x[:, 0:2]], -1)
        self.check(fn, (torch.randn(19, 4).t(),), 1, 0)

    def test_intermediate_and_shared_slice(self):
        def fn(x):
            y = x.sin()
            a = y[:, 1:4]
            return torch.cat([a, y[:, 7:9]], 1), a * 3
        self.check(fn, (torch.randn(5, 12),), 1, 0)

    def test_mixed_non_slice_and_multiple_runs(self):
        def fn(x, y):
            return torch.cat([x[:, 1:2], x[:, 4:7], y, x[:, 8:9], x[:, 0:3]], 1)
        self.check(fn, (torch.randn(5, 12), torch.randn(5, 2)), 2, 0)

    def test_clamp_and_stepped_slice_not_rewritten(self):
        self.check(lambda x: torch.cat([x[:, :2].clamp(-1, 1), x[:, 3:5].clamp(-1, 1)], 1),
                   (torch.randn(4, 12),), 0, 0)
        self.check(lambda x: torch.cat([x[:, :8:2], x[:, 3:9:2]], 1), (torch.randn(4, 12),), 0, 0)

    def test_row_cat_and_empty_slice(self):
        self.check(lambda x: torch.cat([x[:2], x[3:5]], 0), (torch.randn(6, 12),), 0, 0)
        self.check(lambda x: torch.cat([x[:, 7:2], x[:, :2]], 1), (torch.randn(4, 12),), 0, 0)

    def test_two_cast_rounding_steps_preserved(self):
        def fn(x):
            return torch.cat([torch.ops.prims.convert_element_type.default(
                torch.ops.prims.convert_element_type.default(x[:, a:b], torch.float16), torch.float32
            ) for a, b in [(0, 2), (3, 5)]], 1)
        self.check(fn, (torch.randn(4, 12),), 0, 0)

    def test_mask_and_multiple_sources_keep_legacy_route(self):
        def fn(x, y, mask):
            return torch.cat([torch.where(mask, torch.full((4, 2), 0.),
                torch.ops.aten.slice.Tensor(x, 1, 0, 2)),
                torch.ops.aten.slice.Tensor(y, 1, 3, 6)], 1)
        self.check(fn, (torch.randn(4, 12), torch.randn(4, 12), torch.tensor([[True], [False], [True], [False]])), 0, 1, aclnn_copy_free_inputs=0)

    def test_native_mixed_dtype_promotion(self):
        self.check(lambda x, y: torch.cat([x[:, :2], x[:, 3:5], y], 1),
                   (torch.randn(4, 12, dtype=torch.bfloat16), torch.randn(4, 2)), 1, 0)

    def test_unmatched_71_clamp_branches_keep_original_cat(self):
        gm = self.check(lambda x: torch.cat([
            torch.ops.aten.slice.Tensor(x, 1, i, i + 1).clamp(-100, 100)
            for i in range(71)], 1), (torch.randn(4, 80),), 0, 0)
        cats = [n for n in gm.graph.nodes if n.target is torch.ops.aten.cat.default]
        self.assertEqual([len(n.args[0]) for n in cats], [71])

    def test_partial_fusion_keeps_more_than_eight_remaining_inputs(self):
        def fn(x):
            slices = [torch.ops.aten.slice.Tensor(x, 1, i, i+1) for i in range(10)]
            others = [torch.ops.aten.slice.Tensor(x, 1, i, i+1).sin() for i in range(10, 19)]
            return torch.cat(slices + others, 1)
        gm = self.check(fn, (torch.randn(4, 24),), 1, 0)
        cats = [n for n in gm.graph.nodes if n.target is torch.ops.aten.cat.default]
        self.assertEqual([len(n.args[0]) for n in cats], [10])


    @parametrize("ascend950", [False, True])
    def test_native_cats_stay_unmatched(self, ascend950):
        from torch._inductor import config

        slice_op = torch.ops.aten.slice.Tensor
        with patch.object(ascend_graph_pass, "is_ascend950", ascend950), \
                config.patch(max_pointwise_cat_inputs=8):
            for count in (3, 8, 9, 26, 32, 33, 35):
                with self.subTest(count=count):
                    matched = int((ascend950 and count <= 8) or count > 32)
                    gm = self.check(lambda x: torch.cat([slice_op(x, 1, i, i + 1)
                                    for i in range(count)], 1), (torch.randn(4, 40),), matched, 0,
                                    aclnn_copy_free_inputs=32)
                    if not matched:
                        self.assertEqual(sum(n.target is torch.ops.aten.cat.default
                                             for n in gm.graph.nodes), 1)

    @parametrize("ascend950", [False, True])
    @parametrize("pointwise_limit", [4, 6, 8])
    def test_native_gate_follows_backend_and_pass_limits(self, ascend950, pointwise_limit):
        from torch._inductor import config

        def fn(x):
            return torch.cat([x[:, i:i + 1] for i in range(6)], 1)
        with patch.object(ascend_graph_pass, "is_ascend950", ascend950), \
                config.patch(max_pointwise_cat_inputs=pointwise_limit):
            matched = int(ascend950 and 6 <= pointwise_limit)
            self.check(fn, (torch.randn(4, 20),), matched, 0, aclnn_copy_free_inputs=6)
            self.check(fn, (torch.randn(4, 20),), 1, 0, aclnn_copy_free_inputs=5)

    def test_partial_groups_stay_matched(self):
        slice_op = torch.ops.aten.slice.Tensor
        with patch.object(ascend_graph_pass, "is_ascend950", False):
            self.check(lambda x, y: torch.cat([slice_op(x, 1, 0, 2), slice_op(x, 1, 3, 5),
                       y, slice_op(x, 1, 6, 8), slice_op(x, 1, 9, 11)], 1),
                       (torch.randn(4, 20), torch.randn(4, 2)), 2, 0,
                       aclnn_copy_free_inputs=32)

    def test_cast_run_stays_matched(self):
        def fn(x):
            return torch.cat([torch.ops.prims.convert_element_type.default(
                torch.ops.aten.slice.Tensor(x, 1, start, start + 2), torch.float32
            ) for start in (0, 3)], 1)

        with patch.object(ascend_graph_pass, "is_ascend950", False):
            gm = self.check(fn, (torch.randn(4, 20, dtype=torch.bfloat16),), 1, 0,
                            aclnn_copy_free_inputs=32)
        fused = next(n for n in gm.graph.nodes if n.target is self.ns['INDEXED_SLICE_CONCAT_TARGET'])
        self.assertEqual(fused.args[3], torch.float32)

    def test_upstream_row_concat_retains_legacy_route(self):
        self.check(lambda x: torch.cat([torch.ops.aten.slice.Tensor(x, 1, 0, 2),
                   torch.ops.aten.slice.Tensor(x, 1, 3, 5)], 0),
                   (torch.randn(4, 12),), 0, 1)

    def test_upstream_integer_dtype_retains_legacy_route(self):
        self.check(lambda x: torch.cat([torch.ops.aten.slice.Tensor(x, 1, i, i+1)
                   for i in range(35)], 1), (torch.arange(160).reshape(4, 40),), 0, 1)


@instantiate_parametrized_tests
@unittest.skipUnless(hasattr(torch, "npu") and torch.npu.is_available(), "requires NPU")
class TestIndexedSliceConcatDevice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from torch_npu.utils._dynamo import _InductorNpuRegistry
        from torch_npu._inductor.kernel.multi_slice_concat import (
            _register_npu_inductor_multi_slice_concat,
        )
        _InductorNpuRegistry.register_inductor_npu()
        _register_npu_inductor_multi_slice_concat()

    @parametrize("layout", ["plain", "offset", "strided", "transpose"])
    @parametrize("promote", [False, True])
    def test_internal_view_offsets_and_strides(self, layout, promote):
        from torch._inductor import config
        from torch._inductor.utils import run_and_get_code

        x = (torch.arange(640, dtype=torch.float32).reshape(8, 80) / 4).to(
            device="npu", dtype=torch.bfloat16
        )
        torch._dynamo.reset()
        matches = []

        def fn(src):
            if layout == "offset":
                src = src[2:6, 5:65]
            elif layout == "strided":
                src = src[1:7:2, 3:79:2]
            elif layout == "transpose":
                src = src.t()[3:7, 2:8]
            # More than 32 operands keeps the indexed path active.
            count = src.shape[1]
            parts = [src[:, i % count:i % count + 1] for i in range(35)]
            if promote:
                parts = [part.float() for part in parts]
            return torch.cat(parts, dim=1)

        def apply_pass(graph):
            ascend_graph_pass.multi_slice_concat_pass(graph)
            matches.extend(n for n in graph.nodes
                           if n.target is ascend_graph_pass.INDEXED_SLICE_CONCAT_TARGET)

        expected = fn(x)
        with config.patch(post_grad_custom_post_pass=apply_pass):
            actual, codes = run_and_get_code(
                torch.compile(fn, fullgraph=True, dynamic=False), x
            )
        torch.npu.synchronize()
        self.assertTrue(matches)
        self.assertTrue(any("async_compile.triton" in code for code in codes))
        self.assertFalse(any("= torch.ops.npu_ext.indexed_slice_concat.default("
                             in code for code in codes))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @parametrize("cat_dim", [0, 1])
    @parametrize("masked", [False, True])
    def test_legacy_source_and_mask_offsets(self, cat_dim, masked):
        from torch._inductor.utils import run_and_get_code

        x = (torch.arange(640, dtype=torch.float32).reshape(8, 80) / 4).to(
            device="npu", dtype=torch.bfloat16
        )
        y = x + 256
        mask = (torch.arange(12, device="npu") % 3) == 0
        torch._dynamo.reset()

        def fn(src, other, rows):
            return ascend_graph_pass.MULTI_SLICE_CONCAT_TARGET(
                [src[2:6, 5:65], other[1:5, 7:67]],
                [rows[2:10:2, None]] if masked else [],
                [0, 1], [1, 3], [5, 5],
                [0, -1] if masked else [-1, -1], cat_dim,
            )

        first = x[2:6, 6:11]
        if masked:
            first = torch.where(mask[2:10:2, None], 0, first)
        expected = torch.cat([first, y[1:5, 10:15]], cat_dim)
        actual, codes = run_and_get_code(
            torch.compile(fn, fullgraph=True, dynamic=False), x, y, mask
        )
        torch.npu.synchronize()
        self.assertTrue(any("async_compile.triton" in code for code in codes))
        self.assertFalse(any("= torch.ops.npu_ext.multi_slice_concat.default("
                             in code for code in codes))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
