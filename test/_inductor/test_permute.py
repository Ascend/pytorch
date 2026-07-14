import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


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

    @staticmethod
    def transpose_clone_square(x):
        y = x.view(-1, 80, 80, 8)
        return y.permute(0, 2, 1, 3).clone()

    @parametrize('shape', [(8, 8, 512, 128)])
    @parametrize('dtype', ['float32', 'int32', 'float16', 'bfloat16', 'int64'])
    def test_view_cases(self, shape, dtype):
        a = self._generate_tensor(shape, dtype)
        b = self._generate_tensor(shape, dtype)

        for dim in self._permute_dims:
            std_permute = self.op_calc(a, b, dim)
            compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
            inductor_permute = compiled_op_calc(a, b, dim)

            self.assertEqual(std_permute, inductor_permute, atol=1e-3, rtol=1e-3)

    def test_transpose_clone_square(self):
        x = self._generate_tensor((381, 80, 640), "float32")
        eager = self.transpose_clone_square(x)
        compiled = torch.compile(
            self.transpose_clone_square, backend="inductor", dynamic=False
        )
        actual = compiled(x)
        torch.testing.assert_close(actual, eager, rtol=1e-4, atol=1e-4)

instantiate_parametrized_tests(TestPermute)

if __name__ == "__main__":
    run_tests()
