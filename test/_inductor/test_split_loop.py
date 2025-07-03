import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


class TestSplitLoop(TestUtils):
    def op_calc(self, a, b):
        return torch.nn.functional.gelu(a + b)

    @parametrize('shape', [(8, 86, 1152), (61, 89, 157), (7, 89, 971)])
    @parametrize('dtype', ['float32'])
    def test_split_loop(self, shape, dtype):

        a = self._generate_tensor(shape, dtype)
        b = self._generate_tensor((shape[0], 1, shape[2]), dtype)

        std_ = self.op_calc(a, b)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor", dynamic=False)
        inductor_ = compiled_op_calc(a, b)
        self.assertEqual(std_, inductor_, atol=1e-3, rtol=1e-3)


instantiate_parametrized_tests(TestSplitLoop)

if __name__ == "__main__":
    run_tests()
