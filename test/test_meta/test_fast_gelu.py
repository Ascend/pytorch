import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

fake_mode = FakeTensorMode()


class TestFastGelu(TestCase):

    def test_fast_gelu(self):
        with fake_mode:
            a = torch.randn(2, 3).npu()
            a.requires_grad = True
            result = torch_npu.fast_gelu(a)
            self.assertTrue(a.shape == result.shape)

    def test_fast_gelu_backward(self):
        with fake_mode:
            a = torch.randn(2, 3).npu()
            a.requires_grad = True
            result = torch_npu.fast_gelu(a)
            result.sum().backward()
            self.assertTrue(a.shape == a.grad.shape)


if __name__ == "__main__":
    run_tests()
