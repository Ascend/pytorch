import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu
torch._dynamo.config.capture_scalar_outputs = True


class TestAssertScalar(TestUtils):
    def op_calc(self, x):
        condition = x.shape[0] > 0
        torch.ops.aten._assert_scalar(condition, "batch size must be positive")
        return x * 2

    @parametrize('shape', [(512, 64), (4, 32, 128)])
    @parametrize('dtype', ['float32', 'float16'])
    def test_assert_scalar_pass(self, shape, dtype):
        input_element = self._generate_tensor(shape, dtype, floatPOSIFLAG=1)

        std_result = self.op_calc(input_element)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(input_element)

        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)

    def op_calc_with_shape_check(self, x):
        batch_size = x.shape[0]
        condition = batch_size > 0
        torch.ops.aten._assert_scalar(condition, "batch size must be positive")
        return x.mean(dim=0)

    @parametrize('shape', [(8, 64), (16, 128, 32)])
    @parametrize('dtype', ['float32', 'bfloat16'])
    def test_assert_scalar_shape_check(self, shape, dtype):
        input_element = self._generate_tensor(shape, dtype)

        std_result = self.op_calc_with_shape_check(input_element)

        compiled_op_calc = torch.compile(self.op_calc_with_shape_check, backend="inductor")
        inductor_result = compiled_op_calc(input_element)

        self.assertEqual(std_result, inductor_result, atol=1e-3, rtol=1e-3)


instantiate_parametrized_tests(TestAssertScalar)

if __name__ == "__main__":
    run_tests()