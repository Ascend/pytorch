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
import sys
import copy
import torch.nn as nn
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

#TODO:The accuracy of the operator is not up to standard
class TestThnnConvDepthwise2d(TestCase):
    weight_grad = []
    input_grad = []

    def getWeightGrad(self, grad):
        self.weight_grad.append(grad.to("cpu"))

    def getInputGrad(self, grad):
        self.input_grad.append(grad.to("cpu"))

    def op_exec_cpu(self, input, weight, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1,
                    bias=True):
        input1 = input
        weight1 = weight
        input1.requires_grad = True
        input1.register_hook(lambda grad: self.getInputGrad(grad))

        bias1 = False
        if bias != None:
            bias1 = True

        m1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias1,
                       groups=in_channels)
        m1.weight.data = weight1
        m1.weight.register_hook(lambda grad: self.getWeightGrad(grad))
        cpuOutput = m1(input1)
        tmp = torch.ones_like(cpuOutput)
        cpuOutput.backward(tmp)

        return cpuOutput

    def op_exec_npu(self, input, weight, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1,
                    bias=True):
        input1 = input
        weight1 = weight
        input1.requires_grad = True
        input1.register_hook(lambda grad: self.getInputGrad(grad))

        bias1 = False
        if bias != None:
            bias1 = True

        m1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias1,
                       groups=in_channels)
        m1.weight.data = weight1
        m1.weight.register_hook(lambda grad: self.getWeightGrad(grad))
        m1 = m1.to("npu")
        npuOutput = m1(input1)
        npuOutput = npuOutput.to("cpu")
        tmp = torch.ones_like(npuOutput)
        npuOutput.backward(tmp)

        return npuOutput

    def test_conv_depthwise2d_backward_shape_format(self, device):
        shape_format = [  # input, weight, padding, stride, dilation, bias
            [[np.float32, 0, [256, 32, 112, 112]], [np.float32, 0, [32, 1, 3, 3]], 0, 1, 1, True],
            [[np.float32, 0, [256, 32, 112, 112]], [np.float32, 0, [32, 1, 3, 3]], 0, 1, 1, None],
            [[np.float32, 0, [1024, 116, 28, 28]], [np.float32, 0, [116, 1, 3, 3]], 1, [2, 2], 1, True],
            [[np.float32, 0, [1024, 232, 14, 14]], [np.float32, 0, [232, 1, 3, 3]], 1, [2, 2], 1, None],
            [[np.float32, 3, [1024, 232, 7, 7]], [np.float32, 0, [232, 1, 3, 3]], 1, 1, 1, True],
            [[np.float32, 3, [1024, 24, 56, 56]], [np.float32, 0, [24, 1, 3, 3]], 1, [2, 2], 1, None],
            [[np.float32, 3, [1024, 116, 28, 28]], [np.float32, 0, [116, 1, 3, 3]], 1, [2, 2], 1, True],
            [[np.float32, 3, [1024, 232, 14, 14]], [np.float32, 0, [232, 1, 3, 3]], 1, [2, 2], 1, None],
            [[np.float16, 0, [256, 32, 112, 112]], [np.float16, 0, [32, 1, 3, 3]], 0, 1, 1, True],
            [[np.float16, 0, [256, 32, 112, 112]], [np.float16, 0, [32, 1, 3, 3]], 0, 1, 1, None],
            [[np.float16, 0, [1024, 116, 28, 28]], [np.float16, 0, [116, 1, 3, 3]], 1, [2, 2], 1, True],
            [[np.float16, 0, [1024, 232, 14, 14]], [np.float16, 0, [232, 1, 3, 3]], 1, [2, 2], 1, None],
            [[np.float16, 3, [1024, 232, 7, 7]], [np.float16, 0, [232, 1, 3, 3]], 1, 1, 1, True],
            [[np.float16, 3, [1024, 24, 56, 56]], [np.float16, 0, [24, 1, 3, 3]], 1, [2, 2], 1, None],
            [[np.float16, 3, [1024, 116, 28, 28]], [np.float16, 0, [116, 1, 3, 3]], 1, [2, 2], 1, True],
            [[np.float16, 3, [1024, 232, 14, 14]], [np.float16, 0, [232, 1, 3, 3]], 1, [2, 2], 1, None],         
        ]

        for item in shape_format:
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
                                          padding=item[2], stride=item[3], dilation=item[4], bias=item[5])
            weight_npu = weight_npu.to("cpu")
            npu_output = self.op_exec_npu(input_npu, weight_npu, item[0][2][1], item[1][2][0], kernel_size=kernel_size,
                                          padding=item[2], stride=item[3], dilation=item[4], bias=item[5])
            cpu_output = cpu_output.to(npu_output.dtype)
            self.input_grad[0] = self.input_grad[0].to(self.input_grad[1].dtype)
            self.weight_grad[0] = self.weight_grad[0].to(self.weight_grad[1].dtype)

            self.assertRtolEqual(self.input_grad[0].numpy(), self.input_grad[1].numpy())
            self.assertRtolEqual(self.weight_grad[0].numpy(), self.weight_grad[1].numpy())


instantiate_device_type_tests(TestThnnConvDepthwise2d, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
