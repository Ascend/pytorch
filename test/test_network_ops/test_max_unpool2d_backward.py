import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestMaxunpool2dBackward(TestCase):
    def test_maxunpool2d_backward(self, device="npu"):
        input1 = torch.tensor([[[[1., 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]])
        pool2d = torch.nn.MaxPool2d(2, stride=2, return_indices=True)
        out, ind = pool2d(input1)
        unpool2d = torch. nn.MaxUnpool2d(2, stride=2)
        npu_upinput = out.npu()
        npu_ind = ind.npu()
        npu_upinput.requires_grad = True
        out.requires_grad = True
        npu_out = unpool2d(npu_upinput, npu_ind)
        npu_out.backward(torch.ones_like(npu_out))
        npu_grad = npu_upinput.grad
        cpu_out = unpool2d(out, ind)
        cpu_out.backward(torch.ones_like(cpu_out))
        cpu_grad = out.grad
        self.assertRtolEqual(cpu_grad, npu_grad.cpu())

        cpu_out = unpool2d(out, ind)
        grad_input = torch.randn(cpu_out.shape)
        cpu_out.backward(grad_input)
        cpu_grad = out.grad
        npu_out = unpool2d(npu_upinput, npu_ind)
        npu_out.backward(grad_input.npu())
        npu_grad = npu_upinput.grad
        self.assertRtolEqual(cpu_grad, npu_grad.cpu())


if __name__ == "__main__":
    run_tests()
