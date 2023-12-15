# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION.
# All rights reserved.
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

    def op_exec_cpu(self, input1, weight1, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1,
                    bias=True):
        input1.requires_grad = True
        input1.register_hook(lambda grad: self.get_input_grad(grad))

        bias1 = False
        if bias is not None:
            bias1 = True

        m1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias1,
                       groups=in_channels)
        m1.weight.data = weight1
        m1.weight.register_hook(lambda grad: self.get_weight_grad(grad))
        cpuOutput = m1(input1)
        tmp = torch.ones_like(cpuOutput)
        cpuOutput.backward(tmp)

        return cpuOutput

    def op_exec_npu(self, input1, weight1, in_channels, out_channels, kernel_size,
                    padding=0, stride=1, dilation=1, bias=True):
        input1.requires_grad = True
        input1.register_hook(lambda grad: self.get_input_grad(grad))

        bias1 = False
        if bias is not None:
            bias1 = True

        m1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias1,
                       groups=in_channels)
        m1.weight.data = weight1
        m1.weight.register_hook(lambda grad: self.get_weight_grad(grad))
        m1 = m1.to("npu")
        npuOutput = m1(input1)
        npuOutput = npuOutput.to("cpu")
        tmp = torch.ones_like(npuOutput)
        npuOutput.backward(tmp)

        return npuOutput

    def conv_depthwise2d_backward_result(self, shape_format):
        for item in shape_format:
            print(item)
            self.weight_grad.clear()
            self.input_grad.clear()
            input_cpu, input_npu = create_common_tensor(item[0], -1, 1)
            if input_cpu.dtype == torch.float16:
                input_cpu = input_cpu.to(torch.float32)
            weight_cpu, weight_npu = create_common_tensor(item[1], -1, 1)
            if weight_cpu.dtype == torch.float16:
                weight_cpu = weight_cpu.to(torch.float32)
            kernel_size = (item[1][2][2], item[1][2][3])
            cpu_output = self.op_exec_cpu(input_cpu, weight_cpu, item[0][2][1], item[1][2][0], kernel_size=kernel_size,
                                          padding=item[2], stride=item[3], dilation=item[4], bias=item[5])
            npu_output = self.op_exec_npu(input_npu, weight_npu, item[0][2][1], item[1][2][0], kernel_size=kernel_size,
                                          padding=item[2], stride=item[3], dilation=item[4], bias=item[5])
            cpu_output = cpu_output.to(npu_output.dtype)
            self.input_grad[0] = self.input_grad[0].to(self.input_grad[1].dtype)
            self.weight_grad[0] = self.weight_grad[0].to(self.weight_grad[1].dtype)

            if (item[0][0] == np.float32):
                self.assertRtolEqual(self.input_grad[0].numpy(), self.input_grad[1].numpy(), prec=1e-2)
                self.assertRtolEqual(self.weight_grad[0].numpy(), self.weight_grad[1].numpy(), prec=1e-1)
            else:
                self.assertRtolEqual(self.input_grad[0].numpy(), self.input_grad[1].numpy(), prec16=1e-3)
                self.assertRtolEqual(self.weight_grad[0].numpy(), self.weight_grad[1].numpy(), prec16=1e-2)

    def test_conv_depthwise2d_backward_shape_format_fp16(self, device="npu"):
        shape_format = [  # input, weight, padding, stride, dilation, bias
            [[np.float16, 0, [32, 32, 112, 112]], [np.float16, 0, [32, 1, 3, 3]], 0, 1, 1, True],
            [[np.float16, 0, [128, 232, 14, 14]], [np.float16, 0, [232, 1, 3, 3]], 1, [2, 2], 1, None],
            [[np.float16, 3, [128, 232, 7, 7]], [np.float16, 0, [232, 1, 3, 3]], 1, 1, 1, True],
            [[np.float16, 3, [128, 24, 56, 56]], [np.float16, 0, [24, 1, 3, 3]], 1, [2, 2], 1, None],
        ]
        self.conv_depthwise2d_backward_result(shape_format)

    def test_conv_depthwise2d_backward_shape_format_fp32(self, device="npu"):
        shape_format = [  # input, weight, padding, stride, dilation, bias
            [[np.float32, 3, [32, 32, 12, 12]], [np.float32, 0, [32, 1, 3, 3]], 1, 1, 1, None],
            [[np.float32, 0, [32, 32, 12, 12]], [np.float32, 0, [32, 1, 3, 3]], 0, 1, 1, None],
            [[np.float32, 0, [128, 232, 14, 14]], [np.float32, 0, [232, 1, 3, 3]], 1, [2, 2], 1, None],
            [[np.float32, 3, [128, 24, 56, 56]], [np.float32, 0, [24, 1, 3, 3]], 1, [2, 2], 1, None],
        ]
        self.conv_depthwise2d_backward_result(shape_format)


if __name__ == "__main__":
    np.random.seed(1234)
    run_tests()
