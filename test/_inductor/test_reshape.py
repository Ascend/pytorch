import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu




class TestReshape(TestUtils):
    B, N, S, D = (1, 12, 256, 8)

    def op_calc(self, a, b):
        a = a.reshape(self.S, self.B, self.N * self.D)
        b = b.reshape(self.S, self.B, self.N * self.D)
        y = a + b
        return y

    @parametrize('shape', [(1, 12, 256, 8)])
    @parametrize('dtype', ['float32', 'int32', 'float16', 'bfloat16', 'int64'])
    def test_view_cases(self, shape, dtype):
        a = self._generate_tensor(shape, dtype)
        b = self._generate_tensor(shape, dtype)

        std_reshape = self.op_calc(a, b)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_reshape = compiled_op_calc(a, b)

        self.assertEqual(std_reshape, inductor_reshape, atol=1e-3, rtol=1e-3)


instantiate_parametrized_tests(TestReshape)

if __name__ == "__main__":
    run_tests()
