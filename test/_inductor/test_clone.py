import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


class TestClone(TestUtils):
    def op_calc(self, input_element, dim):
        return torch.clone(input_element)

    @parametrize('shape', [(8, 64, 128)])
    @parametrize('dim', [0])
    @parametrize('dtype', ['float32'])
    def test_reduction_cases_shapes(self, shape, dim, dtype):
        input_element = self._generate_tensor(shape, dtype)
        std_ret = self.op_calc(input_element, dim)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_ret = compiled_op_calc(input_element, dim)

        self.assertEqual(std_ret, inductor_ret, equal_nan=True)


instantiate_parametrized_tests(TestClone)

if __name__ == "__main__":
    run_tests()
