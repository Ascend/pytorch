from types import SimpleNamespace

import sympy
from torch._inductor import ir
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_npu  # noqa: F401
from torch_npu._inductor.codegen import triton as triton_codegen


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
        kernel.permute_continous_reduction = False
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
        kernel.permute_continous_reduction = True
        self.assertFalse(kernel._use_contiguous_sum_layout())

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


if __name__ == "__main__":
    run_tests()
