import torch
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from testutils import OperatorType, TestUtils
import torch_npu
import torch_npu._inductor


class TestSumAdd(TestUtils):

    def foo(self, a, b, dim, shape):
        y = a + b
        y = y.sum(dim)
        y = y.unsqueeze(dim)
        y = y.broadcast_to(shape) + b
        return y

    # case：change shapes
    @parametrize('shape', [(9, 9, 31, 63)])
    @parametrize('dim', [0, 1, 2])
    @parametrize('dtype', ['float32'])
    def test_reduction_cases_shapes1(self, shape, dim, dtype):
        a, b = [torch.randn(shape, requires_grad=False, dtype=torch.float32, device="npu") for _ in range(2)]
        r1 = self.foo(a, b, dim, shape)
        func = torch.compile(self.foo, backend="inductor", dynamic=False)
        r = func(a, b, dim, shape)
        torch.testing.assert_close(r, r1, rtol=1e-3, atol=1e-3)

instantiate_parametrized_tests(TestSumAdd)

if __name__ == "__main__":
    run_tests()
