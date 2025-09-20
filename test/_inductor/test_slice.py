import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


class TestSlice(TestUtils):
    def op_calc(self, a, b, dim, step):
        if dim == 0:
            target = a.shape[0]
            end = target // step
            a = a[:end:, ::, ::, ::]
            b = b[:end:, ::, ::, ::]
        elif dim == 1:
            target = a.shape[1]
            end = target // step
            a = a[::, :end:, ::, ::]
            b = b[::, :end:, ::, ::]
        elif dim == 2:
            target = a.shape[2]
            end = target // step
            a = a[::, ::, :end:, ::]
            b = b[::, ::, :end:, ::]
        elif dim == 3:
            target = a.shape[3]
            end = target // step
            a = a[::, ::, ::, :end:]
            b = b[::, ::, ::, :end:]
        y = a + b
        return y

    @parametrize('shape', [(8, 8, 256, 128)])
    @parametrize('dtype', ['float32', 'int32', 'float16', 'bfloat16', 'int64'])
    def test_view_cases(self, shape, dtype):
        a = self._generate_tensor(shape, dtype)
        b = self._generate_tensor(shape, dtype)

        for dim in [3, 2, 1, 0]:
            std_slice = self.op_calc(a, b, dim, min(shape) // 2)

            compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
            inductor_slice = compiled_op_calc(a, b, dim, min(shape) // 2)

            self.assertEqual(std_slice, inductor_slice, atol=1e-3, rtol=1e-3)


instantiate_parametrized_tests(TestSlice)

if __name__ == "__main__":
    run_tests()
