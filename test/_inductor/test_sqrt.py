import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


class TestSqrt(TestUtils):
    def op_calc(self, first_element):
        result = torch.sqrt(first_element)
        return result

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', ['float16', 'float32', 'bfloat16', 'int32', 'int64'])
    def test_pointwise_cases(self, shape, dtype):
        first_element = self._generate_tensor(shape, dtype, 1)

        std_result = self.op_calc(first_element)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element)
        self.assertEqual(std_result, inductor_result, atol=1e-1, rtol=1e-1, equal_nan=True)


instantiate_parametrized_tests(TestSqrt)

if __name__ == "__main__":
    run_tests()
