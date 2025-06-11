import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import OperatorType, TestUtils
import torch_npu
import torch_npu._inductor


class TestRenorm(TestUtils):
    
    def op_calc(self, input_element, dim):
        return torch.renorm(input_element, p=2, dim=dim, maxnorm=5)

    # case：change shapes
    @parametrize('shape', [(32, 64)])
    @parametrize('dim', [-1])
    @parametrize('dtype', ['float32'])
    def test_reduction_cases_shapes(self, shape, dim, dtype):

        input_element = self._generate_tensor(shape, dtype)
        std_ret = self.op_calc(input_element, dim)
        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_ret = compiled_op_calc(input_element, dim)

        torch.testing.assert_close(std_ret, inductor_ret, equal_nan=True)


instantiate_parametrized_tests(TestRenorm)

if __name__ == "__main__":
    run_tests()
