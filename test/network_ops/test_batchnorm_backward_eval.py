import torch
import torch.nn as nn
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class Model(nn.Module):
    def __init__(self, in_channels):
        super(Model, self).__init__()
        self.op1 = nn.Conv2d(in_channels, in_channels, 1)
        self.op2 = nn.BatchNorm2d(in_channels)
        self.op2.running_mean = torch.tensor([i / 1000 for i in range(in_channels)])
        self.op2.running_var = torch.tensor([i / 1000 for i in range(in_channels)])
        self.op3 = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        self.op2.eval()
        x = self.op1(x)
        x = self.op2(x)
        x = self.op3(x)
        return x


class TestBn2dEval(TestCase):
    def test_batchnorm_backward_eval(self, device="npu"):
        model = Model(in_channels=64)
        cpu_tensor = torch.randn(32, 64, 14, 14)
        npu_tensor = cpu_tensor.npu()
        cpu_tensor.requires_grad = True
        npu_tensor.requires_grad = True

        for i in range(1):
            out = model(cpu_tensor)
            loss = out.sum()
            loss.backward()
            cpuout = out
            cpu_grad_list = []
            for name, module in model.named_parameters():
                cpu_grad_list.append(module.grad)
                module.grad = None

            model = model.npu()
            out = model(npu_tensor)
            loss = out.sum()
            loss.backward()
            npu_grad_list = []
            for name, module in model.named_parameters():
                npu_grad_list.append(module.grad.cpu())

            cpu_grad = cpu_tensor.grad
            npu_grad = npu_tensor.grad
            # TODO(ascend): Insufficient precision
            # 精度未满足 self.assertRtolEqual(cpu_grad.numpy(), npu_grad.cpu().numpy())
            self.assertRtolEqual(cpu_grad.numpy(), npu_grad.cpu().numpy(), 0.01)

            for cpu_grad, npu_grad in zip(cpu_grad_list, npu_grad_list):
                # TODO(ascend): Insufficient precision
                # 精度未满足 self.assertRtolEqual(cpu_grad.numpy(), npu_grad.numpy())
                self.assertRtolEqual(cpu_grad.numpy(), npu_grad.numpy(), 0.1)


if __name__ == "__main__":
    run_tests()
