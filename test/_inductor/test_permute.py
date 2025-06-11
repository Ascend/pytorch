import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import OperatorType, TestUtils
import torch_npu
import torch_npu._inductor

torch_npu._inductor.config.enable_npu_indexing = True


class TestPermute(TestUtils):
    
    _permute_dims = [
        (0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3), (0, 2, 3, 1),
        (0, 3, 1, 2), (0, 3, 2, 1), (1, 0, 2, 3), (1, 0, 3, 2),
        (1, 2, 0, 3), (1, 2, 3, 0), (1, 3, 0, 2), (1, 3, 2, 0),
        (2, 0, 1, 3), (2, 0, 3, 1), (2, 1, 0, 3), (2, 1, 3, 0),
        (2, 3, 0, 1), (2, 3, 1, 0), (3, 0, 1, 2), (3, 0, 2, 1),
        (3, 1, 0, 2), (3, 1, 2, 0), (3, 2, 0, 1), (3, 2, 1, 0),
    ]

    def op_calc(self, a, b, dim):
        a = a.permute(dim)
        b = b.permute(dim)
        y = a + b
        return y

    @parametrize('shape', [(8, 8, 512, 128)])
    @parametrize('dtype', ['float32', 'int32', 'float16', 'bfloat16', 'int64'])
    def test_view_cases(self, shape, dtype):
        a = self._generate_tensor(shape, dtype)
        b = self._generate_tensor(shape, dtype)

        for dim in self._permute_dims:
            std_permute = self.op_calc(a, b, dim)
            compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
            inductor_permute = compiled_op_calc(a, b, dim)

            torch.testing.assert_close(std_permute, inductor_permute, rtol=1e-3, atol=1e-3)

instantiate_parametrized_tests(TestPermute)

if __name__ == "__main__":
    run_tests()
