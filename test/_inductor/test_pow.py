import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


class TestPow(TestUtils):
    def op_calc(self, first_element, second_element):
        result = torch.pow(first_element, second_element)
        return result

    @parametrize('shape', [(32, 32),(1, 16, 32)])
    @parametrize('dtype', ['float64', 'float32'])
    def test_pointwise_cases_tensor_tensor(self, shape, dtype):
        first_element =  torch.randn(size=shape, dtype=eval('torch.' + dtype), device=torch.device("npu")) * 2000
        second_element = torch.randn(size=shape, dtype=eval('torch.' + dtype), device=torch.device("npu")) * 2000

        std_result = self.op_calc(first_element, second_element)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element, second_element)
        self.assertEqual(std_result, inductor_result)

    @parametrize('shape', [(32, 32),(1, 16, 32)])
    @parametrize('dtype', ['float64', 'float32'])
    def test_pointwise_cases_scalar_tensor(self, shape, dtype):
        first_tensor =  torch.tensor(100, dtype=eval('torch.' + dtype), device='npu')
        first_element = first_tensor.item()
        second_element = torch.randn(size=shape, dtype=eval('torch.' + dtype), device=torch.device("npu")) * 2000

        std_result = self.op_calc(first_element, second_element)
        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element, second_element)

        self.assertEqual(std_result, inductor_result)

    @parametrize('shape', [(32, 32),(1, 16, 32)])
    @parametrize('dtype', ['float64', 'float32'])
    def test_pointwise_cases_tensor_scalar(self, shape, dtype):
        first_element =  torch.randn(size=shape, dtype=eval('torch.' + dtype), device=torch.device("npu")) * 2000
        second_tensor = torch.tensor(100, dtype=eval('torch.' + dtype), device='npu')
        second_element = second_tensor.item()

        std_result = self.op_calc(first_element, second_element)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element, second_element)
        self.assertEqual(std_result, inductor_result)


instantiate_parametrized_tests(TestPow)

if __name__ == "__main__":
    run_tests()
