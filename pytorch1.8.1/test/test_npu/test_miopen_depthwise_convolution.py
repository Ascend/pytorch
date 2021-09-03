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
import sys
import copy
import torch.nn as nn
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestMiopenDepthwiseConvolution(TestCase):


    def op_exec_cpu(self, input, weight, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1, bias=True):
        input1 = input
        weight1 = weight

        bias1 = False
        if bias != None:
            bias1 = True

        m1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias1, groups=in_channels)
        m1.weight.data = weight1

        cpuOutput = m1(input1)
        tmp = torch.ones_like(cpuOutput)

        return cpuOutput

    def op_exec_npu(self, input, weight, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1, bias=True):
        input1 = input
        weight1 = weight


        bias1 = False
        if bias != None:
            bias1 = True

        m1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias1, groups=in_channels)
        m1.weight.data = weight1
        m1 = m1.to("npu")
        npuOutput = m1(input1)
        npuOutput = npuOutput.to("cpu")
        tmp = torch.ones_like(npuOutput)
        return npuOutput
            
    def test_miopen_depthwise_convolution_input_range1(self, device):
        shape_format = [  # input, weight, padding, stride, dilation, bias
           [[np.float16, 3, [4, 3, 5, 5]], [np.float16, 0, [3, 1, 2, 2]], 0, 1, 1, None],
        ]

        for item in shape_format:

            input_cpu, input_npu = create_common_tensor(item[0],-65504.0,65504.0)
            input_cpu1, input_npu1 = create_common_tensor(item[0],-0.000030517578125,0.000030517578125)
            input_cpu2, input_npu2 = create_common_tensor(item[0],-3402823500.0,3402823500.0)
            input_cpu3, input_npu3 = create_common_tensor(item[0],-0.001953125,0.001953125)
            
            if input_cpu.dtype == torch.float16:
                input_cpu = input_cpu.to(torch.float32)
            weight_cpu, weight_npu = create_common_tensor(item[1], 0,1 )
            if weight_cpu.dtype == torch.float16:
                weight_cpu = weight_cpu.to(torch.float32)

            if input_cpu1.dtype == torch.float16:
                input_cpu1 = input_cpu1.to(torch.float32)
            weight_cpu1, weight_npu1 = create_common_tensor(item[1], 0,1 )
            if weight_cpu1.dtype == torch.float16:
                weight_cpu1 = weight_cpu1.to(torch.float32)
            
            if input_cpu2.dtype == torch.float16:
                input_cpu2 = input_cpu2.to(torch.float32)
            weight_cpu2, weight_npu2 = create_common_tensor(item[1], 0,1 )
            if weight_cpu2.dtype == torch.float16:
                weight_cpu2 = weight_cpu2.to(torch.float32)

            if input_cpu3.dtype == torch.float16:
                input_cpu3 = input_cpu3.to(torch.float32)
            weight_cpu3, weight_npu3 = create_common_tensor(item[1], 0,1 )
            if weight_cpu3.dtype == torch.float16:
                weight_cpu3= weight_cpu3.to(torch.float32)


            kernel_size = (item[1][2][2], item[1][2][3])
            cpu_output = self.op_exec_cpu(input_cpu, weight_cpu, item[0][2][1], item[1][2][0], kernel_size=kernel_size, 
                                          padding=item[2], stride=item[3], dilation=item[4], bias=item[5])
            weight_npu = weight_npu.to("cpu")
            npu_output = self.op_exec_npu(input_npu, weight_npu, item[0][2][1], item[1][2][0], kernel_size=kernel_size,
                                          padding=item[2], stride=item[3], dilation=item[4], bias=item[5])
            cpu_output = cpu_output.to(npu_output.dtype)


            cpu_output1 = self.op_exec_cpu(input_cpu1, weight_cpu1, item[0][2][1], item[1][2][0], kernel_size=kernel_size, 
                                          padding=item[2], stride=item[3], dilation=item[4], bias=item[5])
            weight_npu1 = weight_npu1.to("cpu")
            npu_output1 = self.op_exec_npu(input_npu1, weight_npu1, item[0][2][1], item[1][2][0], kernel_size=kernel_size,
                                          padding=item[2], stride=item[3], dilation=item[4], bias=item[5])
            cpu_output1 = cpu_output1.to(npu_output1.dtype)


            cpu_output2 = self.op_exec_cpu(input_cpu2, weight_cpu2, item[0][2][1], item[1][2][0], kernel_size=kernel_size, 
                                          padding=item[2], stride=item[3], dilation=item[4], bias=item[5])
            weight_npu2 = weight_npu2.to("cpu")
            npu_output2 = self.op_exec_npu(input_npu2, weight_npu2, item[0][2][1], item[1][2][0], kernel_size=kernel_size,
                                          padding=item[2], stride=item[3], dilation=item[4], bias=item[5])
            cpu_output2 = cpu_output2.to(npu_output2.dtype)


            cpu_output3 = self.op_exec_cpu(input_cpu3, weight_cpu3, item[0][2][1], item[1][2][0], kernel_size=kernel_size, 
                                          padding=item[2], stride=item[3], dilation=item[4], bias=item[5])
            weight_npu3 = weight_npu3.to("cpu")
            npu_output3 = self.op_exec_npu(input_npu3, weight_npu3, item[0][2][1], item[1][2][0], kernel_size=kernel_size,
                                          padding=item[2], stride=item[3], dilation=item[4], bias=item[5])
            cpu_output3 = cpu_output3.to(npu_output3.dtype)


            print("===========cpu_output============")
            print(cpu_output)
            print("===========cpu_output1============")
            print(cpu_output1)
            print("===========cpu_output2============")
            print(cpu_output2)
            print("===========cpu_output3============")
            print(cpu_output3)

            print("===========npu_output============")
            print(npu_output)
            print("===========npu_output1============")
            print(npu_output1)
            print("===========npu_output2============")
            print(npu_output2)
            print("===========npu_output3============")
            print(npu_output3)


            print("===========cpu_input&&npu_input==================")
            print(input_cpu)
            
            self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().numpy())

    def test_miopen_depthwise_convolution_shape_format(self, device):
        shape_format = [  # input, weight, padding, stride, dilation, bias
            [[np.float16, 3, [256, 32, 112, 112]], [np.float16, 0, [32, 1, 3, 3]], 0, 1, 1, None],
            [[np.float16, 0, [1024, 116, 28, 28]], [np.float16, 0, [116, 1, 3, 3]], 1, [2, 2], 1, None],
            [[np.float16, 0, [1024, 232, 14, 14]], [np.float16, 0, [232, 1, 3, 3]], 1, [2, 2], 1, None],
            [[np.float16, 3, [1024, 232, 7, 7]], [np.float16, 0, [232, 1, 3, 3]], 1, 1, 1, None],
            [[np.float16, 3, [1024, 24, 56, 56]], [np.float16, 0, [24, 1, 3, 3]], 1, [2, 2], 1, None],
            [[np.float16, 3, [1024, 116, 28, 28]], [np.float16, 0, [116, 1, 3, 3]], 1, [2, 2], 1, None],
            [[np.float16, 3, [1024, 232, 14, 14]], [np.float16, 0, [232, 1, 3, 3]], 1, [2, 2], 1, None],
        ]

        for item in shape_format:

            input_cpu, input_npu = create_common_tensor(item[0],-65504.0,65504.0)
            if input_cpu.dtype == torch.float16:
                input_cpu = input_cpu.to(torch.float32)
            weight_cpu, weight_npu = create_common_tensor(item[1], 0,1 )
            if weight_cpu.dtype == torch.float16:
                weight_cpu = weight_cpu.to(torch.float32)
            kernel_size = (item[1][2][2], item[1][2][3])
            cpu_output = self.op_exec_cpu(input_cpu, weight_cpu, item[0][2][1], item[1][2][0], kernel_size=kernel_size, 
                                          padding=item[2], stride=item[3], dilation=item[4], bias=item[5])
            weight_npu = weight_npu.to("cpu")
            npu_output = self.op_exec_npu(input_npu, weight_npu, item[0][2][1], item[1][2][0], kernel_size=kernel_size,
                                          padding=item[2], stride=item[3], dilation=item[4], bias=item[5])
            cpu_output = cpu_output.to(npu_output.dtype)

            print("===========cpu_output============")
            print(cpu_output)

            print("===========npu_output============")
            print(npu_output)

            print("===========cpu_input&&npu_input==================")
            print(input_cpu)
            
            self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().numpy())



instantiate_device_type_tests(TestMiopenDepthwiseConvolution, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
