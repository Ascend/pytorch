import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


class TestClamp(TestUtils):

    def op_calc(self, arg, min_value=None, max_value=None):
        return arg.clamp(min_value, max_value)

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', ['float16', 'float32', 'bfloat16', 'int32', 'int64'])
    def test_pointwise_cases_minmax_is_tensor(self, shape, dtype):
        min_0 = self._generate_tensor(shape, dtype)
        max_0 = self._generate_tensor(shape, dtype)

        first_element = self._generate_tensor(shape, dtype)

        std_result = self.op_calc(first_element, min_value=min_0, max_value=max_0)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element, min_value=min_0, max_value=max_0)

        self.assertEqual(std_result, inductor_result)

    @parametrize('shape', [(1,)])
    @parametrize('dtype', ['float32'])
    def test_pointwise_cases_single_scalar(self, shape, dtype):
        min_numel = 0
        max_numel = 100

        first_element = 200 * torch.rand(size=shape, dtype=eval('torch.' + dtype), device=torch.device("npu"))

        std_result = self.op_calc(first_element, min_value=min_numel, max_value=max_numel)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element, min_value=min_numel, max_value=max_numel)
        self.assertEqual(std_result, inductor_result)

    @parametrize('shape', [(1024, 32)])
    @parametrize('dtype', ['int32'])
    def test_pointwise_cases_minmax_is_number(self, shape, dtype):
        min_numel = 0
        max_numel = 100

        first_element = self._generate_tensor(shape, dtype)

        std_result = self.op_calc(first_element, min_value=min_numel, max_value=max_numel)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element, min_value=min_numel, max_value=max_numel)

        self.assertEqual(std_result, inductor_result)

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', ['float16', 'float32', 'bfloat16', 'int32', 'int64'])
    def test_pointwise_cases_max_only(self, shape, dtype):
        max_numel = 100

        first_element = self._generate_tensor(shape, dtype)

        std_result = self.op_calc(first_element, min_value=None, max_value=max_numel)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element, min_value=None, max_value=max_numel)

        self.assertEqual(std_result, inductor_result)

    @parametrize('shape', TestUtils._pointwise_demo_shapes)
    @parametrize('dtype', ['float16', 'float32', 'bfloat16', 'int32', 'int64'])  
    def test_pointwise_cases_min_only(self, shape, dtype):
        min_numel = 0

        first_element = self._generate_tensor(shape, dtype)

        std_result = self.op_calc(first_element, min_value=min_numel, max_value=None)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element, min_value=min_numel, max_value=None)

        self.assertEqual(std_result, inductor_result)


instantiate_parametrized_tests(TestClamp)

if __name__ == "__main__":
    run_tests()
