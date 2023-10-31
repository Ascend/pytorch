import torch
import numpy as np
import torch.nn as nn
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestConv3d(TestCase):
    weight_grad = []
    input_grad = []

    def get_weight_grad(self, grad):
        self.weight_grad.append(grad.to("cpu"))

    def get_input_grad(self, grad):
        self.input_grad.append(grad.to("cpu"))

    def op_exec_cpu(self, x, weight, in_channels, out_channels,
                    kernel_size, padding=0, stride=1, dilation=1, bias=True, groups=1):
        input1 = x
        weight1 = weight
        input1.requires_grad = True
        input1.register_hook(lambda grad: self.get_input_grad(grad))

        m1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias, groups=groups)
        m1.weight.data = weight1
        m1.weight.register_hook(lambda grad: self.get_weight_grad(grad))
        cpuOutput = m1(input1)
        tmp = torch.ones_like(cpuOutput)
        cpuOutput.backward(tmp)
        return cpuOutput

    def op_exec_npu(self, x, weight, in_channels, out_channels, kernel_size,
                    padding=0, stride=1, dilation=1, bias=False, groups=1):
        input1 = x
        weight1 = weight
        input1.requires_grad = True
        input1.register_hook(lambda grad: self.get_input_grad(grad))

        m1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias, groups=groups)
        m1.weight.data = weight1
        m1.weight.register_hook(lambda grad: self.get_weight_grad(grad))
        m1 = m1.to("npu")
        npuOutput = m1(input1)
        tmp = torch.ones_like(npuOutput)
        npuOutput.backward(tmp)

        return npuOutput.to("cpu")

    def conv3d_backward_result(self, shape_format):
        for item in shape_format:
            self.weight_grad.clear()
            self.input_grad.clear()
            np.random.seed(1234)
            input_cpu, input_npu = create_common_tensor(item[0], 0, 1)
            if input_cpu.dtype == torch.float16:
                input_cpu = input_cpu.to(torch.float32)
            weight_cpu, weight_npu = create_common_tensor(item[1], 0, 1)
            if weight_cpu.dtype == torch.float16:
                weight_cpu = weight_cpu.to(torch.float32)
            kernel_size = (item[1][2][2], item[1][2][3], item[1][2][4])
            cpu_output = self.op_exec_cpu(input_cpu, weight_cpu, item[0][2][1],
                                          item[1][2][0], kernel_size=kernel_size,
                                          padding=item[2], stride=item[3],
                                          dilation=item[4], bias=item[5], groups=item[6])
            weight_npu = weight_npu.to("cpu")

            npu_output = self.op_exec_npu(input_npu, weight_npu, item[0][2][1],
                                          item[1][2][0], kernel_size=kernel_size,
                                          padding=item[2], stride=item[3],
                                          dilation=item[4], bias=item[5], groups=item[6])

            npu_output = npu_output.to(torch.float16)
            cpu_output = cpu_output.to(torch.float16)
            self.input_grad[0] = self.input_grad[0].to(torch.float16)
            self.input_grad[1] = self.input_grad[1].to(torch.float16)
            self.weight_grad[0] = self.weight_grad[0].to(self.weight_grad[1].dtype)
            self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.cpu().detach().numpy(), 1e-3)
            self.assertRtolEqual(self.input_grad[0].numpy(), self.input_grad[1].cpu().numpy(), 1e-3)
            self.assertRtolEqual(self.weight_grad[0].numpy(), self.weight_grad[1].cpu().numpy(), 1e-3)

    def test_conv3d_backward_shape_format_fp16(self):
        shape_format = [  # input, weight, padding, stride, dilation, bias, groups
            [[np.float16, 30, [1, 128, 4, 14, 14]],
             [np.float16, 30, [1, 128, 3, 3, 3]], [1, 1, 1], [1, 1, 1], 1, None, 1],
            [[np.float16, 30, [1, 64, 4, 14, 14]],
             [np.float16, 30, [1, 64, 3, 3, 3]], [1, 1, 1], [2, 2, 2], 1, None, 1],
        ]
        self.conv3d_backward_result(shape_format)

    def test_conv3d_backward_shape_format_fp32(self):
        shape_format = [  # input, weight, padding, stride, dilation, bias, groups
            [[np.float32, 30, [1, 128, 4, 14, 14]],
             [np.float32, 30, [1, 128, 3, 3, 3]], [1, 1, 1], [1, 1, 1], 1, None, 1],
            [[np.float32, 30, [1, 64, 4, 14, 14]],
             [np.float32, 30, [1, 64, 3, 3, 3]], [1, 1, 1], [2, 2, 2], 1, None, 1],
            [[np.float32, 32, [1, 64, 8, 28, 28]],
             [np.float32, 33, [1, 64, 3, 3, 3]], [1, 1, 1], [1, 1, 1], [1, 1, 1], None, 1]
        ]
        self.conv3d_backward_result(shape_format)


if __name__ == "__main__":
    run_tests()
