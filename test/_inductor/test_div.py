import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


class TestDiv(TestUtils):
    def op_calc(self, first_element, second_element):
        result = torch.div(first_element, second_element)
        return result

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', ['float16', 'float32', 'bfloat16', 'int32', 'int64'])
    def test_pointwise_cases(self, shape, dtype):
        first_element = self._generate_tensor(shape, dtype)
        second_element = self._generate_tensor(shape, dtype)

        std_result = self.op_calc(first_element, second_element)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element, second_element)
        self.assertEqual(std_result, inductor_result)


instantiate_parametrized_tests(TestDiv)

if __name__ == "__main__":
    run_tests()
