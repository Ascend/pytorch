import torch
from torch.testing._internal.common_utils import (
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)
from testutils import TestUtils
import torch_npu  # noqa: F401


class TestInplaceOpCompile(TestUtils):
    """Regression tests for compiling graphs that contain in-place operators.

    NPUCombinedScheduling initialises through BaseScheduling instead of
    CUDACombinedScheduling, so the NV universal-GEMM scheduler is never created.
    Inherited methods that still reach for it raised AttributeError as soon as
    the scheduler evaluated a reduction fusion pair, which in-place operators
    trigger.  The affected methods are can_fuse_reduction_pair,
    can_fuse_reduction_epilogue and get_fusion_pair_priority in
    torch_npu/_inductor/codegen/npu_combined_scheduling.py.
    """

    def _run_case(self, op_calc, shape, dtype):
        element = self._generate_tensor(shape, dtype)
        expected_input = element.clone()

        compiled_op_calc = torch.compile(op_calc, backend="inductor")
        result = compiled_op_calc(element)
        expected_result = op_calc(expected_input)

        # 原地语义：入参必须被改写，且结果与 eager 一致
        self.assertEqual(element, expected_input)
        self.assertEqual(result, expected_result)

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', ['float32'])
    def test_add_inplace(self, shape, dtype):
        self._run_case(lambda x: x.add_(1.0), shape, dtype)

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', ['float32'])
    def test_mul_inplace_with_epilogue(self, shape, dtype):
        self._run_case(lambda x: x.mul_(2.0) + 1.0, shape, dtype)

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', ['float32'])
    def test_relu_inplace_with_epilogue(self, shape, dtype):
        self._run_case(lambda x: x.relu_() * 2.0, shape, dtype)


instantiate_parametrized_tests(TestInplaceOpCompile)

if __name__ == "__main__":
    run_tests()
