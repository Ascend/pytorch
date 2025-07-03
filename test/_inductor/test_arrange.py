import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


class TestArrange(TestUtils):
    def op_calc(self, start, end, step):
        a = torch.arange(start, end, step, device=torch.device('npu'))
        y = a + a
        return y

    @parametrize('shape', [(2, )])
    @parametrize('dtype', TestUtils._test_dtypes)
    def test_pointwise_cases(self, shape, dtype):
        s = self._generate_tensor(shape, dtype)
        start = min(s)
        end = max(s)
        step = (end - start) / 32

        std_arrange = self.op_calc(start, end, step)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor", dynamic=False)
        inductor_arrange = compiled_op_calc(start, end, step)

        self.assertEqual(std_arrange, inductor_arrange)

instantiate_parametrized_tests(TestArrange)

if __name__ == "__main__":
    run_tests()
