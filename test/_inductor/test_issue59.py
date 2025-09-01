import torch
from torch.testing._internal.common_utils import run_tests
from testutils import TestUtils
import torch_npu


class Test_issue59(TestUtils):
    def layernorm_backward(self, x, y, z):
        sum_0 = torch.sum(x)
        mean = sum_0 / torch.numel(sum_0)
        sub = x - mean
        sqr = sub * sub
        sum_1 = torch.sum(sqr)
        mean_1 = sum_1 / torch.numel(sum_1) + 1e-05
        rsqrt = torch.rsqrt(mean_1)
        mul = sub * rsqrt
        mul_1 = mul * y
        add = mul_1 + z
        mean_2 = rsqrt / torch.numel(rsqrt)
        return mul, add, mean_2

    def test_issue59(self):
        device = 'npu'
        x = torch.randn((1, 1024), device=device, dtype=torch.float32)
        y = torch.randn((1, 1024), device=device, dtype=torch.float32)
        z = torch.randn((1, 1024), device=device, dtype=torch.float32)

        mul, add, mean_2 = self.layernorm_backward(x, y, z)
        func = torch.compile(self.layernorm_backward, backend="inductor", dynamic=False)
        mul_t, add_t, mean_2_t = func(x, y, z)

        self.assertEqual(mul, mul_t, atol=1e-3, rtol=1e-3)
        self.assertEqual(add, add_t, atol=1e-3, rtol=1e-3)
        self.assertEqual(mean_2, mean_2_t, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    run_tests()
