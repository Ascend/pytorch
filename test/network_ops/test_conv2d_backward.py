import sys
import numpy as np

import torch
import torch.nn as nn

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestConv2dBackward(TestCase):
    weight_grad = []
    input_grad = []

    def getWeightGrad(self, grad):
        self.weight_grad.append(grad.to("cpu"))

    def getInputGrad(self, grad):
        self.input_grad.append(grad.to("cpu"))

    def op_exec(self, npu_flag, input1, weight, in_channels, out_channels, kernel_size, padding=0, stride=1,
                dilation=1, bias=True, groups=1):
        weight1 = weight
        input1.requires_grad = True
        input1.register_hook(lambda grad: self.getInputGrad(grad))

        m1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias, groups=groups)
        m1.weight.data = weight1
        m1.weight.register_hook(lambda grad: self.getWeightGrad(grad))
        if npu_flag:
            m1 = m1.to("npu")
        output = m1(input1)
        tmp = torch.ones_like(output)
        output.backward(tmp)
        if npu_flag:
            output = output.to("cpu")
        return output

    def test_conv2d_backward_shape_format_fp16(self):
        shape_format = [  # input, weight, padding, stride, dilation, bias, groups
            # shuflenet
            [[np.float16, 3, [1024, 232, 7, 7]], [np.float16, 4, [232, 232, 1, 1]], 0, 1, 1, None, 1],
            [[np.float16, 0, [1024, 116, 14, 14]], [np.float16, 4, [116, 116, 1, 1]], 0, 1, 1, None, 1],
            [[np.float16, 0, [4, 8, 300, 40]], [np.float16, 0, [16, 8, 3, 3]], [2, 1], 1, 1, None, 1],
            [[np.float16, 0, [4, 64, 150, 10]], [np.float16, 0, [32, 64, 1, 1]], 0, 1, 1, None, 1],
            [[np.float16, 0, [4, 128, 75, 10]], [np.float16, 0, [64, 128, 1, 1]], 0, 1, 1, None, 1],
            [[np.float16, 0, [4, 256, 75, 5]], [np.float16, 0, [128, 256, 3, 3]], [2, 1], 1, 1, None, 1],
            [[np.float16, 0, [4, 384, 75, 1]], [np.float16, 0, [192, 384, 3, 1]], 0, 1, 1, None, 1],
            [[np.float16, 0, [4, 384, 1, 75]], [np.float16, 0, [192, 384, 1, 3]], 0, 1, 1, None, 1],
            [[np.float16, 3, [4, 256, 75, 5]], [np.float16, 4, [128, 256, 3, 3]], [2, 1], 1, 1, None, 1],
            [[np.float16, 3, [4, 384, 75, 1]], [np.float16, 4, [192, 384, 3, 1]], 0, 1, 1, None, 1],
            [[np.float16, 3, [4, 384, 1, 75]], [np.float16, 4, [192, 384, 1, 3]], 0, 1, 1, None, 1],
            [[np.float16, 0, [4, 256, 75, 5]], [np.float16, 4, [128, 256, 3, 3]], [2, 1], 1, 1, None, 1],
            [[np.float16, 0, [4, 384, 75, 1]], [np.float16, 4, [192, 384, 3, 1]], 0, 1, 1, None, 1],
            [[np.float16, 0, [4, 384, 1, 75]], [np.float16, 4, [192, 384, 1, 3]], 0, 1, 1, None, 1]
        ]
        for item in shape_format:
            self.weight_grad.clear()
            self.input_grad.clear()
            input_cpu, input_npu = create_common_tensor(item[0], -1, 1)
            if input_cpu.dtype == torch.float16:
                input_cpu = input_cpu.to(torch.float32)
            weight_cpu, weight_npu = create_common_tensor(item[1], -1, 1)
            if weight_cpu.dtype == torch.float16:
                weight_cpu = weight_cpu.to(torch.float32)
            kernel_size = (item[1][2][2], item[1][2][3])
            if item[0][2][1] / item[6] != item[1][2][1]:
                raise ValueError("ilegal parameters: con2d in_channels//groups must equal to weight.size[1].")
            cpu_output = self.op_exec(0, input_cpu, weight_cpu, item[0][2][1], item[1][2][0], kernel_size=kernel_size,
                                      padding=item[2], stride=item[3], dilation=item[4], bias=item[5],
                                      groups=item[6])
            weight_npu = weight_npu.to("cpu")
            npu_output = self.op_exec(1, input_npu, weight_npu, item[0][2][1], item[1][2][0], kernel_size=kernel_size,
                                      padding=item[2], stride=item[3], dilation=item[4], bias=item[5],
                                      groups=item[6])

            npu_output = npu_output.to(torch.float16)
            cpu_output = cpu_output.to(torch.float16)
            self.input_grad[0] = self.input_grad[0].to(torch.float16)
            self.input_grad[1] = self.input_grad[1].to(torch.float16)

            self.weight_grad[0] = self.weight_grad[0].to(self.weight_grad[1].dtype)

            self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().numpy())
            self.assertRtolEqual(self.input_grad[0].numpy(), self.input_grad[1].numpy())
            self.assertRtolEqual(self.weight_grad[0].numpy(), self.weight_grad[1].numpy())


if __name__ == "__main__":
    run_tests()
