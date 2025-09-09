import copy
import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import TestUtils
import torch_npu


class TestBroadcast(TestUtils):
    broadcast_size = 128

    def op_calc(self, a, b, dim, new_shape):
        a = a.unsqueeze(dim)
        a = a.broadcast_to(new_shape)
        b = b.unsqueeze(dim)
        b = b.broadcast_to(new_shape)
        y = a + b
        return y


    @parametrize('shape', [(8, 8, 256)])
    @parametrize('dtype', ['float32', 'int32', 'float16', 'bfloat16'])
    def test_view_cases(self, shape, dtype):
        a = self._generate_tensor(shape, dtype)
        b = self._generate_tensor(shape, dtype)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        for dim in [3, 2, 1, 0]:
            new_shape = list(copy.deepcopy(shape))
            new_shape.insert(dim, self.broadcast_size)
            std_broadcast = self.op_calc(a, b, dim, new_shape)
            inductor_broadcast = compiled_op_calc(a, b, dim, new_shape)

            self.assertEqual(std_broadcast.float(), inductor_broadcast.float(), atol=1e-3, rtol=1e-3)


instantiate_parametrized_tests(TestBroadcast)

if __name__ == "__main__":
    run_tests()
