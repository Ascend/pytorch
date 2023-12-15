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


class TestThnnConvDepthwise2d(TestCase):
    weight_grad = []
    input_grad = []

    def get_weight_grad(self, grad):
        self.weight_grad.append(grad.to("cpu"))

    def get_input_grad(self, grad):
        self.input_grad.append(grad.to("cpu"))

    def op_exec_cpu(self, input1, weight, in_channels,
                    out_channels, kernel_size, padding=0, stride=1, dilation=1, bias=True, group=2):
        weight1 = weight
        input1.requires_grad = True
        input1.register_hook(lambda grad: self.get_input_grad(grad))

        bias1 = False
        if bias is not None:
            bias1 = True

        m1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias1, groups=group)
        m1.weight.data = weight1
        m1.weight.register_hook(lambda grad: self.get_weight_grad(grad))
        cpuOutput = m1(input1)
        cpuOutput = cpuOutput.requires_grad_()
        tmp = torch.ones_like(cpuOutput)
        cpuOutput.backward(tmp)

        return cpuOutput

    def op_exec_npu(self, input1, weight, in_channels,
                    out_channels, kernel_size, padding=0, stride=1, dilation=1, bias=True, group=2):
        weight1 = weight
        input1.requires_grad = True
        input1.register_hook(lambda grad: self.get_input_grad(grad))

        bias1 = False
        if bias is not None:
            bias1 = True

        m1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias1, groups=group)
        m1.weight.data = weight1
        m1.weight.register_hook(lambda grad: self.get_weight_grad(grad))
        m1 = m1.to("npu")
        npuOutput = m1(input1)
        npuOutput = npuOutput.to("cpu")
        npuOutput = npuOutput.requires_grad_()
        tmp = torch.ones_like(npuOutput)
        npuOutput.backward(tmp)

        return npuOutput

    def thnn_conv_depthwise2d_format(self, i):
        shape_format = [  # input, weight, padding, stride, dilation, bias
            [[np.float32, 3, (64, 3, 32, 32)], [np.float32, -1, (3, 1, 3, 3)], 0, 1, (1, 1), True],
            [[np.float16, 3, (128, 3, 64, 64)], [np.float16, -1, (3, 1, 3, 3)], 0, 1, 1, None],
            [[np.float16, 3, (32, 3, 16, 16)], [np.float16, -1, (3, 1, 3, 3)], 0, 1, 1, None],
            [[np.float16, 3, (32, 6, 32, 32)], [np.float16, -1, (6, 1, 3, 3)], 0, 1, 1, None],
            [[np.float16, 3, (32, 6, 32, 32)], [np.float16, -1, (6, 1, 3, 3)], 0, 1, 1, None]
        ]
        return shape_format[i]

    def thnn_conv_depthwise2d_execute(self, item, group):
        self.weight_grad.clear()
        self.input_grad.clear()
        input_cpu, input_npu = create_common_tensor(item[0], 0, 10)
        if input_cpu.dtype == torch.float16:
            input_cpu = input_cpu.to(torch.float32)
        weight_cpu, weight_npu = create_common_tensor(item[1], 0, 10)
        if weight_cpu.dtype == torch.float16:
            weight_cpu = weight_cpu.to(torch.float32)
        kernel_size = (item[1][2][2], item[1][2][3])
        cpu_output = self.op_exec_cpu(input_cpu, weight_cpu, item[0][2][1], item[1][2][0], kernel_size=kernel_size,
                                      padding=item[2], stride=item[3], dilation=item[4], bias=item[5], group=group)
        weight_npu = weight_npu.to("cpu")
        npu_output = self.op_exec_npu(input_npu, weight_npu, item[0][2][1], item[1][2][0], kernel_size=kernel_size,
                                      padding=item[2], stride=item[3], dilation=item[4], bias=item[5], group=group)
        cpu_output = cpu_output.to(npu_output.dtype)

        if item[5] is True:
            self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().numpy(), 0.005)
        else:
            self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().numpy())

    def test_thnn_conv_depthwise2d_0(self, device="npu"):
        item = self.thnn_conv_depthwise2d_format(0)
        self.thnn_conv_depthwise2d_execute(item, 3)

    def test_thnn_conv_depthwise2d_1(self, device="npu"):
        item = self.thnn_conv_depthwise2d_format(1)
        self.thnn_conv_depthwise2d_execute(item, 3)

    def test_thnn_conv_depthwise2d_2(self, device="npu"):
        item = self.thnn_conv_depthwise2d_format(2)
        self.thnn_conv_depthwise2d_execute(item, 3)

    def test_thnn_conv_depthwise2d_3(self, device="npu"):
        item = self.thnn_conv_depthwise2d_format(3)
        self.thnn_conv_depthwise2d_execute(item, 6)

    def test_thnn_conv_depthwise2d_4(self, device="npu"):
        item = self.thnn_conv_depthwise2d_format(4)
        self.thnn_conv_depthwise2d_execute(item, 6)


if __name__ == "__main__":
    np.random.seed(1234)
    run_tests()
