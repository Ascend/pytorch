import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


class TestMaxWithIndex(TestUtils):
    def op_calc(self, input_element, dim):
        return torch.argmax(input_element, dim)

    @parametrize('shape', [(512, 64)]) # (513, 64), (514,33)
    @parametrize('dim', [-1])
    @parametrize('dtype', ['float32'])
    def test_reduction_cases(self, shape, dim, dtype):
        input_element = torch.randn(size=shape, dtype=eval('torch.' + dtype), device=torch.device("npu")) * 2000
        std_argmax = self.op_calc(input_element, dim)
        compiled_op_calc = torch.compile(self.op_calc, backend="inductor", dynamic=False)
        inductor_argmax = compiled_op_calc(input_element, dim)
        self.assertEqual(std_argmax, inductor_argmax, atol=1e-2, rtol=1e-2)

instantiate_parametrized_tests(TestMaxWithIndex)

if __name__ == "__main__":
    run_tests()
