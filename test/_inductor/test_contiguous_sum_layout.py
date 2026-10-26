import unittest
from types import SimpleNamespace

import sympy
import torch
from torch._inductor import ir
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)

import torch_npu  # noqa: F401
from torch_npu._inductor.codegen import triton as triton_codegen
from version_mark import runIfVersion


class TestContiguousSumLayout(TestCase):
    @staticmethod
    def _kernel(value_names, parsed_names=None, reduction_types=("sum",)):
        names = set(value_names) | set(parsed_names or value_names)
        axes = {name: sympy.Symbol(name) for name in names}
        value_axes = [axes[name] for name in value_names]
        reduction_axes = [axis for axis in value_axes if axis.name.startswith("r")]
        reductions = []
        for reduction_type in reduction_types:
            reduction = object.__new__(ir.Reduction)
            object.__setattr__(reduction, "reduction_type", reduction_type)
            reductions.append(reduction)

        kernel = object.__new__(triton_codegen.NPUIndexTritonKernel)
        kernel.golden_var_list = tuple(reversed(value_axes))
        kernel.tiling_axis = value_axes
        kernel.numof_reduction_axis = lambda: len(reduction_axes)
        kernel.reduction_axis_list = lambda: reduction_axes
        parsed = [axes[name] for name in reversed(parsed_names or value_names)]
        kernel.parse_golden_from_load_store_index = lambda: parsed
        kernel.node_schedule = [SimpleNamespace(node=SimpleNamespace(data=node)) for node in reductions]
        kernel.features = SimpleNamespace(node_schedule=kernel.node_schedule)
        kernel.find_reduction_node = lambda: reductions[0] if reductions else None
        kernel.inside_reduction = True
        kernel.reduce_analysis = None
        kernel.vectorized_welford_axis = None
        return kernel

    def test_outer_scalar_does_not_split_value_reduction_axes(self):
        kernel = self._kernel(("r2", "r3", "x5"), ("r2", "x4", "r3", "x5"))
        self.assertTrue(kernel._use_contiguous_sum_layout())
        self.assertTrue(kernel.is_contiguous_reduction())
        self.assertEqual(kernel.dense_size_str(), "[X5BLOCK_SUB, R3BLOCK_SUB * R2BLOCK_SUB]")
        self.assertEqual(kernel.reduction_dim(), 1)
        self.assertEqual(kernel.triton_tensor_ndim(), 2)
        self.assertEqual(kernel.reduction_resize("s", 1), "s.reshape(1, 1, X5BLOCK_SUB)")

    def test_multiple_preserved_axes_keep_their_order(self):
        kernel = self._kernel(("x0", "r1", "r2", "x3"), ("x0", "r1", "x4", "r2", "x3"))
        self.assertTrue(kernel._use_contiguous_sum_layout())
        self.assertEqual(kernel.dense_size_str(), "[X0BLOCK_SUB, X3BLOCK_SUB, R2BLOCK_SUB * R1BLOCK_SUB]")
        self.assertEqual(kernel.reduction_dim(), 2)
        self.assertEqual(kernel.triton_tensor_ndim(), 3)
        self.assertEqual(kernel.reduction_resize("s", 2), "s.reshape(X0BLOCK_SUB, 1, 1, X3BLOCK_SUB)")

    def test_reduction_axes_do_not_reappear_in_packed_shape_outside_reduction(self):
        kernel = self._kernel(("r2", "r3", "x5"), ("r2", "x4", "r3", "x5"))
        expected = "[X5BLOCK_SUB, R3BLOCK_SUB * R2BLOCK_SUB]"
        self.assertEqual(kernel.dense_size_str(), expected)
        kernel.inside_reduction = False
        self.assertEqual(kernel.reduce_analysis.dense_size_list(), ["1", "1", "X5BLOCK_SUB"])
        self.assertEqual(kernel.dense_size_str(), expected)

    def test_single_separated_and_scalar_reductions_keep_the_old_layout(self):
        for names, size_str, dim, ndim in (
            (("r0", "x1"), "[R0BLOCK_SUB, X1BLOCK_SUB]", 0, 2),
            (("r0", "x1", "r2"), "[R0BLOCK_SUB* X1BLOCK_SUB* R2BLOCK_SUB]", 0, 1),
            (("r0", "r1"), "[R1BLOCK_SUB * R0BLOCK_SUB]", 0, 1),
        ):
            with self.subTest(names=names):
                kernel = self._kernel(names)
                self.assertFalse(kernel._use_contiguous_sum_layout())
                self.assertEqual(kernel.dense_size_str(), size_str)
                self.assertEqual(kernel.reduction_dim(), dim)
                self.assertEqual(kernel.triton_tensor_ndim(), ndim)
        self.assertEqual(kernel.reduction_resize("s", 0), "triton_helpers.promote_to_tensor(s)")

    def test_specialized_and_mixed_reductions_do_not_enable_the_correction(self):
        for types in (
            ("max",), ("min",), ("argmax",), ("argmin",), ("welford_reduce",),
            ("prod",), ("sum", "max"), ("sum", "welford_reduce"), (),
        ):
            with self.subTest(types=types):
                kernel = self._kernel(
                    ("r2", "r3", "x5"), ("r2", "x4", "r3", "x5"), reduction_types=types,
                )
                self.assertFalse(kernel._use_contiguous_sum_layout())
                self.assertFalse(kernel.is_contiguous_reduction())
                self.assertEqual(kernel.triton_tensor_ndim(), 1)

    def test_legacy_adjacent_address_order_is_preserved(self):
        kernel = self._kernel(("r0", "x1", "r2"), ("r0", "r2", "x1"))
        self.assertFalse(kernel._use_contiguous_sum_layout())
        self.assertTrue(kernel.is_contiguous_reduction())

    def test_constructor_uses_the_feature_schedule_before_node_schedule_exists(self):
        kernel = self._kernel(("r2", "r3", "x5"), ("r2", "x4", "r3", "x5"))
        del kernel.node_schedule
        self.assertTrue(kernel._use_contiguous_sum_layout())


@unittest.skipUnless(torch.npu.is_available(), "NPU is not available")
class TestContiguousSumLayoutNPU(TestCase):
    @runIfVersion(max="2.13")
    @parametrize("dynamic", (False, True))
    @parametrize("width", (63, 64))
    @parametrize("consume_result", (False, True))
    def test_bias_gradient_and_broadcast_consumer(self, dynamic, width, consume_result):
        def fn(value):
            grouped = value.reshape(4, 12, 128, width)
            bias = grouped.permute(0, 2, 1, 3).reshape(512, 12 * width).sum(0)
            if consume_result:
                return bias, grouped + bias.reshape(1, 12, 1, width)
            return bias

        torch.manual_seed(0)
        value = torch.randn((48, 128, width), device="npu", dtype=torch.float32)
        expected = fn(value)
        try:
            compiled = torch.compile(fn, backend="inductor", dynamic=dynamic, fullgraph=True)
            actual = compiled(value)
            torch.npu.synchronize()
            self.assertEqual(actual, expected, atol=2e-4, rtol=1e-3)
        finally:
            torch._dynamo.reset()


instantiate_parametrized_tests(TestContiguousSumLayoutNPU)


if __name__ == "__main__":
    run_tests()
