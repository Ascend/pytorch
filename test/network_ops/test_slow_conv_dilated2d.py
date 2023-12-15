# Copyright (c) 2020, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
import torch.nn as nn
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestSlowConvDilated2D(TestCase):

    def cpu_op_exec(self):
        inputs = torch.randn(1, 3, 5, 5, requires_grad=True)
        filters = torch.randn(4, 3, 3, 3, requires_grad=True)
        bias = torch.randn(4, requires_grad=True)
        output = torch.nn.functional.conv2d(inputs, filters, bias, dilation=2)
        output = output.detach().numpy()
        return output

    def npu_op_exec(self):
        inputs = torch.randn(1, 3, 5, 5, requires_grad=True)
        filters = torch.randn(4, 3, 3, 3, requires_grad=True)
        bias = torch.randn(4, requires_grad=True)
        inputs = inputs.to("npu")
        filters = filters.to("npu")
        bias = bias.to("npu")
        output = torch.nn.functional.conv2d(inputs, filters, bias, dilation=2)
        output = output.to("cpu")
        output = output.detach().numpy()
        return output

    weight_grad = []
    input_grad = []

    def getWeightGrad(self, grad):
        self.weight_grad.append(grad.to("cpu"))

    def getInputGrad(self, grad):
        self.input_grad.append(grad.to("cpu"))

    def op_exec_conv2d(self, input2, weight, bias_, in_channels, out_channels,
                       kernel_size, padding, stride, dilation, bias):
        input1 = input2
        weight1 = weight
        input1.requires_grad = True
        input1.register_hook(lambda grad: self.getInputGrad(grad))

        bias1 = False
        if bias is not None:
            bias1 = True

        m1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias1, groups=1)
        m1.weight.data = weight1
        if bias1:
            m1.bias.data = bias_
        m1.weight.register_hook(lambda grad: self.getWeightGrad(grad))
        return input1, m1

    def op_exec_cpu(self, input2, weight, bias_, in_channels, out_channels,
                    kernel_size, padding=0, stride=1, dilation=2, bias=True):
        input1, m1 = self.op_exec_conv2d(input2, weight, bias_, in_channels, out_channels,
                                         kernel_size, padding, stride, dilation, bias)
        cpu_output = m1(input1)
        tmp = torch.ones_like(cpu_output)
        cpu_output.backward(tmp)

        return cpu_output

    def op_exec_npu(self, input2, weight, bias_, in_channels, out_channels,
                    kernel_size, padding=0, stride=1, dilation=2, bias=True):
        input1, m1 = self.op_exec_conv2d(input2, weight, bias_, in_channels, out_channels,
                                         kernel_size, padding, stride, dilation, bias)
        m1 = m1.to("npu")
        npu_output = m1(input1)
        npu_output = npu_output.to("cpu")
        tmp = torch.ones_like(npu_output)
        npu_output.backward(tmp)

        return npu_output

    def test_slow_conv_dilated2d_shape_format_fp16(self):
        self._slow_conv_dilated2d_shape_format(np.float16)

    def test_slow_conv_dilated2d_shape_format_fp32(self):
        self._slow_conv_dilated2d_shape_format(np.float32, 3, 5)

    def _slow_conv_dilated2d_shape_format(self, data_type, start=0, end=None):
        shape_format = [
            [[data_type, 3, [256, 32, 112, 112]], [data_type, 0, [16, 32, 1, 1]],
             0, 1, 2, None],
            [[data_type, 0, [256, 3, 224, 224]], [data_type, 0, [32, 3, 3, 3]],
             0, [2, 2], 2, [data_type, 0, 32]],
            [[data_type, 3, [256, 128, 7, 7]], [data_type, 4, [32, 128, 3, 3]],
             (1, 1), 1, 2, None],
            [[data_type, 3, (2, 3, 3, 3)], [data_type, 0, (3, 3, 3, 3)],
             3, 1, 2, [data_type, 0, 3]],
            [[data_type, 3, [1024, 232, 7, 7]], [data_type, 4, [232, 232, 1, 1]],
             0, 1, 2, [data_type, 4, 232]],
            [[data_type, 0, [1024, 116, 14, 14]], [data_type, 4, [116, 116, 1, 1]],
             0, 1, 2, None],
            [[data_type, 0, [1024, 58, 28, 28]], [data_type, 4, [58, 58, 1, 1]],
             0, 1, 2, [data_type, 4, 58]],
            [[data_type, 3, [1024, 116, 14, 14]], [data_type, 4, [116, 116, 1, 1]],
             0, 1, 2, [data_type, 4, 116]],
            [[data_type, 0, [1024, 232, 7, 7]], [data_type, 4, [232, 232, 1, 1]],
             0, 1, 2, None],
            [[data_type, 3, [1024, 58, 28, 28]], [data_type, 4, [58, 58, 1, 1]],
             0, 1, 2, [data_type, 4, 58]],
        ]

        for item in shape_format[start:end]:
            self.weight_grad.clear()
            self.input_grad.clear()
            input_cpu, input_npu = create_common_tensor(item[0], 0, 10)
            if input_cpu.dtype == torch.float16:
                input_cpu = input_cpu.to(torch.float32)
            weight_cpu, weight_npu = create_common_tensor(item[1], 0, 10)
            if weight_cpu.dtype == torch.float16:
                weight_cpu = weight_cpu.to(torch.float32)
            bias_cpu = bias_npu = None
            if item[5] is not None:
                bias_cpu, bias_npu = create_common_tensor(item[5], 0, 1)
                if bias_cpu.dtype == torch.float16:
                    bias_cpu = bias_cpu.to(torch.float32)
            kernel_size = (item[1][2][2], item[1][2][3])
            cpu_output = self.op_exec_cpu(input_cpu, weight_cpu, bias_cpu,
                                          item[0][2][1], item[1][2][0], kernel_size=kernel_size,
                                          padding=item[2], stride=item[3], dilation=item[4], bias=item[5])
            weight_npu = weight_npu.to("cpu")
            if bias_npu is not None:
                bias_npu = bias_npu.to("cpu")
            npu_output = self.op_exec_npu(input_npu, weight_npu, bias_npu,
                                          item[0][2][1], item[1][2][0], kernel_size=kernel_size,
                                          padding=item[2], stride=item[3], dilation=item[4], bias=item[5])
            cpu_output = cpu_output.to(npu_output.dtype)
            self.input_grad[0] = self.input_grad[0].to(self.input_grad[1].dtype)
            self.weight_grad[0] = self.weight_grad[0].to(self.weight_grad[1].dtype)

            self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().numpy(), 0.001)


if __name__ == "__main__":
    run_tests()
