import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


class TestRepeat(TestUtils):
    def op_calc(self, input_element, dim):
        return input_element.repeat(dim)

    # case：change shapes
    @parametrize('shape', [(16, 128, 64)])
    @parametrize('dim', [(1, 1, 2), (1, 2, 1), (2, 1, 1)])
    @parametrize('dtype', ['float32'])
    def test_reduction_cases_shapes(self, shape, dim, dtype):
        input_element = self._generate_tensor(shape, dtype)

        std_ret = self.op_calc(input_element, dim)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor", dynamic=False)
        inductor_ret = compiled_op_calc(input_element, dim)

        self.assertEqual(std_ret, inductor_ret, atol=1e-1, rtol=1e-1)


instantiate_parametrized_tests(TestRepeat)

if __name__ == "__main__":
    run_tests()
