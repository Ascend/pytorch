import unittest

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from torch_npu._inductor.fx_passes.ascend_custom_passes import ascend_graph_pass as passes


def slices(x, widths=(2, 3, 4)):
    return [torch.ops.aten.slice.Tensor(x, 1, i * 5, i * 5 + width)
            for i, width in enumerate(widths)]


class TestSinkPointwiseCat(unittest.TestCase):

    def check(self, fn, inputs, changed):
        gm = make_fx(fn, tracing_mode='fake')(*inputs)
        expected = gm(*inputs)
        before = str(gm.graph)
        passes.sink_pointwise_cat_pass(gm.graph)
        gm.graph.lint()
        gm.recompile()
        self.assertEqual(str(gm.graph) != before, changed)
        torch.testing.assert_close(gm(*inputs), expected, rtol=0, atol=0, equal_nan=True)
        after = str(gm.graph)
        passes.sink_pointwise_cat_pass(gm.graph)
        self.assertEqual(str(gm.graph), after)
        return gm

    def test_supported_unary(self):
        for op in [lambda x: x.clamp(-1, 1),
                   torch.abs, torch.neg]:
            for widths in [(2, 3, 4), (3, 3, 3)]:
                with self.subTest(op=op, widths=widths):
                    gm = self.check(lambda x: torch.cat([op(t) for t in slices(x, widths)], 1),
                                    (torch.randn(4, 20, dtype=torch.bfloat16),), True)
                    self.assertFalse(any(n.target in (torch.ops.aten.stack.default,
                                                      torch.ops.aten.split_with_sizes.default)
                                         for n in gm.graph.nodes))

    def test_cast_only_keeps_existing_slice_cat_fusion(self):
        gm = self.check(lambda x: torch.cat([
            torch.ops.prims.convert_element_type.default(t, torch.float32) for t in slices(x)], 1),
            (torch.randn(4, 20, dtype=torch.bfloat16),), False)
        passes.multi_slice_concat_pass(gm.graph, aclnn_copy_free_inputs=0)
        self.assertEqual(sum(n.target is passes.INDEXED_SLICE_CONCAT_TARGET for n in gm.graph.nodes), 1)

    def test_clamp_chain_keeps_slice_cat_in_source_dtype(self):

        def fn(x):
            return torch.cat([torch.ops.prims.convert_element_type.default(
                torch.ops.prims.convert_element_type.default(t, torch.float32).clamp(-1, 1),
                torch.bfloat16) for t in slices(x)], 1)
        gm = self.check(fn, (torch.randn(4, 20, dtype=torch.bfloat16),), True)
        passes.multi_slice_concat_pass(gm.graph, aclnn_copy_free_inputs=0)
        fused = [n for n in gm.graph.nodes if n.target is passes.INDEXED_SLICE_CONCAT_TARGET]
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].meta['val'].dtype, torch.bfloat16)

    def test_binary_and_where_broadcast(self):
        for op in [lambda x, y: x + y, lambda x, y: x * y]:
            self.check(lambda x, y: torch.cat([op(t, y) for t in slices(x)], 1),
                       (torch.randn(4, 20), torch.randn(4, 1)), True)
        self.check(lambda x, mask, z: torch.cat([torch.where(mask, t, z) for t in slices(x)], 1),
                   (torch.randn(4, 20), torch.randn(4, 1) > 0, torch.zeros(4, 1)), True)

    def test_per_column_bias_and_different_parameters_are_rejected(self):
        self.check(lambda x, y: torch.cat([t + y for t in slices(x, (3, 3, 3))], 1),
                   (torch.randn(4, 20), torch.randn(4, 3)), False)
        self.check(lambda x: torch.cat([t.clamp(-i, i + 1) for i, t in enumerate(slices(x))], 1),
                   (torch.randn(4, 20),), False)

    def test_common_cast_does_not_hide_different_inner_operations(self):

        def fn(x):
            a, b, c = slices(x)
            return torch.cat([torch.ops.prims.convert_element_type.default(t, torch.float32)
                              for t in (a.clamp(-1, 1), b.clamp(-2, 2), c.neg())], 1)
        self.check(fn, (torch.randn(4, 20, dtype=torch.bfloat16),), False)

    def test_common_outer_operation_preserves_distinct_chains(self):
        self.check(lambda x, y: torch.cat([t.clamp(-i - 1, i + 1) * y
                   for i, t in enumerate(slices(x))], 1),
                   (torch.randn(4, 20), torch.randn(4, 1)), False)

    def test_only_adjacent_identical_chains_are_combined(self):

        def fn(x):
            parts = [x[:, i:i + 2] for i in range(7)]
            return torch.cat([t.clamp(-1, 1) if i != 3 else t.neg()
                              for i, t in enumerate(parts)], 1)
        gm = self.check(fn, (torch.randn(4, 20),), True)
        self.assertEqual(sum(n.target is torch.ops.aten.clamp.default for n in gm.graph.nodes), 2)
        self.assertEqual(sum(n.target is torch.ops.aten.neg.default for n in gm.graph.nodes), 1)

    def test_multiple_casts_are_not_reordered(self):
        gm = self.check(lambda x: torch.cat([torch.ops.prims.convert_element_type.default(
            torch.ops.prims.convert_element_type.default(t, torch.float16), torch.bfloat16)
            for t in slices(x)], 1), (torch.randn(4, 20),), False)
        casts = [n.args[1] for n in gm.graph.nodes
                 if n.target is torch.ops.prims.convert_element_type.default]
        self.assertEqual(casts, [torch.float16, torch.bfloat16] * 3)

    def test_intermediate_rounding_cast_is_not_reordered(self):
        for op in (torch.abs, lambda t: t.clamp(-1, 1)):
            def fn(x):
                return torch.cat([op(torch.ops.prims.convert_element_type.default(
                    t, torch.bfloat16)).float() for t in slices(x)], 1)

            self.check(fn, (torch.randn(4, 20),), False)

    def test_partial_runs_and_shared_users(self):
        self.check(lambda x, y: torch.cat([t.clamp(-1, 1) for t in slices(x)] + [y], 1),
                   (torch.randn(4, 20), torch.randn(4, 1)), True)

        def fn(x):
            parts = [t.clamp(-1, 1) for t in slices(x)]
            return torch.cat(parts, 1), tuple(parts)
        self.check(fn, (torch.randn(4, 20),), False)

    def test_matmul_and_reduction_are_rejected(self):
        self.check(lambda x, w: torch.cat([(t @ w).clamp(-1, 1)
                   for t in slices(x, (3, 3, 3))], 1),
                   (torch.randn(4, 20), torch.randn(3, 2)), False)
        self.check(lambda x: torch.cat([t.sum(1, keepdim=True).clamp(-1, 1)
                   for t in slices(x)], 1), (torch.randn(4, 20),), False)

    def test_mixed_source_dtype_is_not_reordered(self):
        self.check(lambda x, y: torch.cat([x[:, :2].float(), y[:, :3].float()], 1),
                   (torch.randn(4, 20, dtype=torch.bfloat16), torch.randn(4, 20, dtype=torch.float16)), False)

    def test_slice_cat_is_exposed(self):
        gm = self.check(lambda x: torch.cat([t.clamp(-1, 1) for t in slices(x)], 1),
                        (torch.randn(4, 20),), True)
        passes.multi_slice_concat_pass(gm.graph, aclnn_copy_free_inputs=0)
        self.assertEqual(sum(n.target is passes.INDEXED_SLICE_CONCAT_TARGET for n in gm.graph.nodes), 1)

    def test_single_input_cat_keeps_copy_semantics(self):
        x = torch.randn(4, 20)
        gm = self.check(lambda x: torch.cat([x[:, :3]], 1), (x,), False)
        self.assertNotEqual(gm(x).untyped_storage().data_ptr(), x.untyped_storage().data_ptr())

    def test_two_branches_are_not_reordered(self):
        self.check(lambda x: torch.cat([t.clamp(-1, 1) for t in slices(x, (2, 3))], 1),
                   (torch.randn(4, 20),), False)

    def test_relu_is_not_reordered(self):
        # NPU compiled ReLU currently differs from eager for signed zero.
        self.check(lambda x: torch.cat([t.relu() for t in slices(x)], 1),
                   (torch.randn(4, 20),), False)

    def test_zero_clamp_is_not_reordered(self):
        for op in [lambda t: t.clamp_min(0.), lambda t: t.clamp_max(0.),
                   lambda t: t.clamp(-0., 0.),
                   lambda t: torch.ops.prims.convert_element_type.default(t.clamp_max(0.), torch.float32)]:
            self.check(lambda x: torch.cat([op(t) for t in slices(x)], 1),
                       (torch.randn(4, 20, dtype=torch.bfloat16),), False)

    def test_repeated_branch_preserves_duplicates(self):

        def fn(x):
            part = x[:, :3].clamp(-1, 1)
            return torch.cat([part, part, part], 1)
        self.check(fn, (torch.randn(4, 20),), False)

        def separated(x):
            part = x[:, :3].clamp(-1, 1)
            return torch.cat([part, part, part, x[:, :3].neg(), part, part, part], 1)
        self.check(separated, (torch.randn(4, 20),), False)

    def test_row_cat_and_empty_rows(self):
        self.check(lambda x, y: torch.cat([t + y for t in slices(x, (3, 3, 3))], 0),
                   (torch.randn(4, 20), torch.randn(1, 3)), False)
        self.check(lambda x: torch.cat([t.clamp(-1, 1) for t in slices(x)], 1),
                   (torch.empty(0, 20),), True)

    def test_unsupported_slice_layouts_are_not_reordered(self):
        for fn, inputs in [
            (lambda x: torch.cat([x[:, :, i:i + 2].clamp(-1, 1) for i in range(3)], 2),
             (torch.randn(2, 4, 20),)),
            (lambda x: torch.cat([x[:, i:i + 6:2].clamp(-1, 1) for i in range(3)], 1),
             (torch.randn(4, 20),)),
            (lambda x, y, z: torch.cat([t[:, :3].clamp(-1, 1) for t in (x, y, z)], 1),
             (torch.randn(4, 20), torch.randn(4, 20), torch.randn(4, 20)))]:
            self.check(fn, inputs, False)


@unittest.skipUnless(hasattr(torch, 'npu') and torch.npu.is_available(), 'requires NPU')
class TestSinkPointwiseCatDevice(unittest.TestCase):

    def test_different_chains_keep_original_graph(self):
        from torch._inductor import config

        def fn(x):
            a, b, c = slices(x)
            different = torch.cat([t.float() for t in
                                   (a.clamp(-1, 1), b.clamp(-2, 2), c.neg())], 1)
            shared = a.clamp(-3, 3)
            return different, torch.cat([shared, shared, shared], 1)

        rewrites = []

        def hook(graph):
            before = str(graph)
            passes.sink_pointwise_cat_pass(graph)
            rewrites.append(before != str(graph))
            passes.multi_slice_concat_pass(graph)

        x = torch.randn(4, 40, dtype=torch.bfloat16, device='npu')[:, 1:39:2]
        with config.patch(post_grad_custom_post_pass=hook, fx_graph_cache=False):
            actual = torch.compile(fn, fullgraph=True)(x)
        torch.testing.assert_close(actual, fn(x), rtol=0, atol=0)
        self.assertTrue(rewrites)
        self.assertFalse(any(rewrites))

    def test_compiled_precision_and_special_values(self):
        self.check_kinds(('cast', 'clamp', 'mul', 'where'))

    def test_add_abs_neg(self):
        self.check_kinds(('add', 'abs', 'neg'))

    def check_kinds(self, kinds):
        from torch._inductor import config
        from torch._inductor.utils import run_and_get_code
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for kind in kinds:
                torch._dynamo.reset()

                def fn(x, y):
                    parts = slices(x)
                    if kind == 'cast':
                        parts = [torch.ops.prims.convert_element_type.default(t, torch.float32) for t in parts]
                    elif kind == 'clamp':
                        parts = [t.clamp(-1, 1) for t in parts]
                    elif kind == 'mul':
                        parts = [t * y for t in parts]
                    elif kind == 'add':
                        parts = [t + y for t in parts]
                    elif kind in ('abs', 'neg'):
                        parts = [getattr(torch, kind)(t) for t in parts]
                    elif kind == 'where':
                        mask, z = y > 0, torch.zeros_like(y)
                        parts = [torch.where(mask, t, z) for t in parts]
                    return torch.cat(parts, 1)
                values = [0., -0., float('inf'), -float('inf'), float('nan'), 1.00390625 + 2. ** -15,
                          1.00048828125, -1.00390625, 1.e-8, -1.e-8] * 8
                storage = torch.tensor(values, device='npu', dtype=dtype).reshape(2, 40)
                x = storage[:, 1:39:2]
                y = torch.tensor([[1.], [-1.]], device='npu', dtype=dtype)
                expected = fn(x, y)
                rewrites = []

                def hook(graph):
                    before = str(graph)
                    passes.sink_pointwise_cat_pass(graph)
                    rewrites.append((before != str(graph), before))
                    passes.multi_slice_concat_pass(graph)
                with self.subTest(dtype=dtype, kind=kind), config.patch(
                        post_grad_custom_post_pass=hook, fx_graph_cache=False, fx_graph_remote_cache=False):
                    actual, codes = run_and_get_code(torch.compile(fn, fullgraph=True), x, y)
                    torch.npu.synchronize()
                    if kind != 'cast':
                        self.assertTrue(any(changed for changed, _ in rewrites), str(rewrites))
                    else:
                        self.assertFalse(any(changed for changed, _ in rewrites), str(rewrites))
                    actual, expected = actual.cpu(), expected.cpu()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
                    zeros = expected == 0
                    self.assertTrue(torch.equal(torch.signbit(actual[zeros]), torch.signbit(expected[zeros])))
