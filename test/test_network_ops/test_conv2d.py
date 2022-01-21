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
import torch_npu
import numpy as np
import torch.nn as nn

from torch_npu.testing.common_utils import TestCase, run_tests
from torch_npu.testing.common_device_type import instantiate_device_type_tests
from torch_npu.testing.util_test import create_common_tensor


class TestConv2d(TestCase):
    weight_grad = []
    input_grad = []

    def get_weight_grad(self, grad):
        self.weight_grad.append(grad.to("cpu"))

    def get_input_grad(self, grad):
        self.input_grad.append(grad.to("cpu"))

    def op_exec_cpu(self, x, weight, in_channels, out_channels, kernel_size,
                    padding=0, stride=1, dilation=1, bias=True, groups=1):
        input1 = x
        weight1 = weight
        input1.requires_grad = True
        input1.register_hook(lambda grad: self.get_input_grad(grad))

        m1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias, groups=groups)
        m1.weight.data = weight1
        m1.weight.register_hook(lambda grad: self.get_weight_grad(grad))
        cpuOutput = m1(input1)
        tmp = torch.ones_like(cpuOutput)
        cpuOutput.backward(tmp)

        return cpuOutput

    def op_exec_npu(self, x, weight, in_channels, out_channels, kernel_size,
                    padding=0, stride=1, dilation=1, bias=True, groups=1):
        input1 = x
        weight1 = weight
        input1.requires_grad = True
        input1.register_hook(lambda grad: self.get_input_grad(grad))

        m1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias, groups=groups)
        m1.weight.data = weight1
        m1.weight.register_hook(lambda grad: self.get_weight_grad(grad))
        m1 = m1.to("npu")
        npuOutput = m1(input1)
        tmp = torch.ones_like(npuOutput)
        npuOutput.backward(tmp)

        return npuOutput.to("cpu")

    def conv2d_backward_result(self, shape_format):
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
            assert item[0][2][1]/item[6] == item[1][2][1], "ilegal parameters: con2d in_channels//groups must equal to weight.size[1]."
            cpu_output = self.op_exec_cpu(input_cpu, weight_cpu, item[0][2][1], item[1][2][0], kernel_size=kernel_size,
                                          padding=item[2], stride=item[3], dilation=item[4], bias=item[5], groups=item[6])
            weight_npu = weight_npu.to("cpu")
            npu_output = self.op_exec_npu(input_npu, weight_npu, item[0][2][1], item[1][2][0], kernel_size=kernel_size,
                                          padding=item[2], stride=item[3], dilation=item[4], bias=item[5], groups=item[6])

            npu_output = npu_output.to(torch.float16)
            cpu_output = cpu_output.to(torch.float16)
            self.input_grad[0] = self.input_grad[0].to(torch.float16)
            self.input_grad[1] = self.input_grad[1].to(torch.float16)

            self.weight_grad[0] = self.weight_grad[0].to(self.weight_grad[1].dtype)

            self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().numpy())
            self.assertRtolEqual(self.input_grad[0].numpy(), self.input_grad[1].numpy())
            self.assertRtolEqual(self.weight_grad[0].numpy(), self.weight_grad[1].numpy())

    def test_conv2d_backward_shape_format_fp16(self, device):
        shape_format = [  # input, weight, padding, stride, dilation, bias, groups            
            # shuflenet
            [[np.float16, 3, [1024, 232, 7, 7]], [np.float16, 4, [232, 232, 1, 1]], 0, 1, 1, None, 1],
            [[np.float16, 0, [1024, 116, 14, 14]], [np.float16, 4, [116, 116, 1, 1]], 0, 1, 1, None, 1],
            [[np.float16, 0, [4, 8, 300, 40]], [np.float16, 0, [16, 8, 3, 3]], [2,1], 1, 1, None, 1], 
            [[np.float16, 0, [4, 64, 150, 10]], [np.float16, 0, [32, 64, 1, 1]], 0, 1, 1, None, 1], 
            [[np.float16, 0, [4, 128, 75, 10]], [np.float16, 0, [64, 128, 1, 1]], 0, 1, 1, None, 1], 
            [[np.float16, 0, [4, 256, 75, 5]], [np.float16, 0, [128, 256, 3, 3]], [2,1], 1, 1, None, 1], 
            [[np.float16, 0, [4, 384, 75, 1]], [np.float16, 0, [192, 384, 3, 1]], 0, 1, 1, None, 1], 
            [[np.float16, 0, [4, 384, 1, 75]], [np.float16, 0, [192, 384, 1, 3]], 0, 1, 1, None, 1], 
            [[np.float16, 3, [4, 256, 75, 5]], [np.float16, 4, [128, 256, 3, 3]], [2,1], 1, 1, None, 1], 
            [[np.float16, 3, [4, 384, 75, 1]], [np.float16, 4, [192, 384, 3, 1]], 0, 1, 1, None, 1], 
            [[np.float16, 3, [4, 384, 1, 75]], [np.float16, 4, [192, 384, 1, 3]], 0, 1, 1, None, 1], 
            [[np.float16, 0, [4, 256, 75, 5]], [np.float16, 4, [128, 256, 3, 3]], [2,1], 1, 1, None, 1], 
            [[np.float16, 0, [4, 384, 75, 1]], [np.float16, 4, [192, 384, 3, 1]], 0, 1, 1, None, 1], 
            [[np.float16, 0, [4, 384, 1, 75]], [np.float16, 4, [192, 384, 1, 3]], 0, 1, 1, None, 1], 
            # 当前不支持kernel_size_h >= padding_h*2 + input_h和kernel_size_w >= padding_w*2 + input_w, 预计330支持
            # [[np.float16, 0, [4, 384, 75, 1]], [np.float16, 0, [192, 384, 3, 3]], 0, 1, 1, None, 1],
            # [[np.float16, 0, [4, 384, 75, 1]], [np.float16, 0, [192, 384, 3, 3]], [1,1], 1, 1, None, 1],
            # [[np.float16, 0, [4, 384, 1, 75]], [np.float16, 0, [192, 384, 3, 3]], 0, 1, 1, None, 1],
            # [[np.float16, 0, [4, 384, 1, 75]], [np.float16, 0, [192, 384, 3, 3]], [1,1], 1, 1, None, 1],
        ]
        self.conv2d_backward_result(shape_format)

    def test_conv2d_backward_shape_format_fp32(self, device):
        shape_format = [  # input, weight, padding, stride, dilation, bias, groups            
            # mobilenet
            [[np.float32, 3, [256, 960, 7, 7]], [np.float32, 0, [320, 960, 1, 1]], 0, 1, 1, None, 1],
            [[np.float32, 0, [256, 3, 224, 224]], [np.float32, 0, [32, 3, 3, 3]], 1, 2, 1, None, 1],
            [[np.float32, 0, [16, 3, 640, 640]], [np.float32, 4, [64, 3, 7, 7]], 3, 2, 1, None, 1],
            [[np.float32, 0, [4, 8, 300, 40]], [np.float32, 0, [16, 8, 3, 3]], [2,1], 1, 1, None, 1], 
            [[np.float32, 0, [4, 64, 150, 10]], [np.float32, 0, [32, 64, 1, 1]], 0, 1, 1, None, 1], 
            [[np.float32, 0, [4, 128, 75, 10]], [np.float32, 0, [64, 128, 1, 1]], 0, 1, 1, None, 1], 
            [[np.float32, 0, [4, 256, 75, 5]], [np.float32, 0, [128, 256, 3, 3]], [2,1], 1, 1, None, 1], 
            [[np.float32, 0, [4, 384, 75, 1]], [np.float32, 0, [192, 384, 3, 1]], 0, 1, 1, None, 1], 
            [[np.float32, 0, [4, 384, 1, 75]], [np.float32, 0, [192, 384, 1, 3]], 0, 1, 1, None, 1], 
            [[np.float32, 3, [4, 256, 75, 5]], [np.float32, 0, [128, 256, 3, 3]], [2,1], 1, 1, None, 1], 
            [[np.float32, 3, [4, 384, 75, 1]], [np.float32, 0, [192, 384, 3, 1]], 0, 1, 1, None, 1], 
            [[np.float32, 3, [4, 384, 1, 75]], [np.float32, 0, [192, 384, 1, 3]], 0, 1, 1, None, 1], 
            [[np.float32, 0, [4, 256, 75, 5]], [np.float32, 4, [128, 256, 3, 3]], [2,1], 1, 1, None, 1], 
            [[np.float32, 0, [4, 384, 75, 1]], [np.float32, 4, [192, 384, 3, 1]], 0, 1, 1, None, 1], 
            [[np.float32, 0, [4, 384, 1, 75]], [np.float32, 4, [192, 384, 1, 3]], 0, 1, 1, None, 1], 
            # 当前不支持kernel_size_h >= padding_h*2 + input_h和kernel_size_w >= padding_w*2 + input_w, 预计330支持
            # [[np.float32, 0, [4, 384, 75, 1]], [np.float32, 0, [192, 384, 3, 3]], 0, 1, 1, None, 1],
            # [[np.float32, 0, [4, 384, 75, 1]], [np.float32, 0, [192, 384, 3, 3]], [1,1], 1, 1, None, 1],
            # [[np.float32, 0, [4, 384, 1, 75]], [np.float32, 0, [192, 384, 3, 3]], 0, 1, 1, None, 1],
            # [[np.float32, 0, [4, 384, 1, 75]], [np.float32, 0, [192, 384, 3, 3]], [1,1], 1, 1, None, 1],
            ]
        #conv类算子不支持fp32数据的精度要求
        #self.conv2d_backward_result(shape_format)

    def test_group_conv2d_backward_shape_format_fp16(self, device):
        shape_format = [  # input, weight, padding, stride, dilation, bias, groups
            # KDXF
            [[np.float16, 0, [4, 64, 75, 10]], [np.float16, 0, [128, 16, 3, 3]], [2,1], 1, 1, None, 4],
            [[np.float16, 0, [4, 128, 75, 10]], [np.float16, 0, [64, 32, 1, 1]], 0, 1, 1, None, 4],
            [[np.float16, 0, [4, 128, 75, 5]], [np.float16, 0, [256, 32, 3, 3]], [2,1], 1, 1, None, 4],
            [[np.float16, 0, [4, 256, 75, 1]], [np.float16, 0, [384, 64, 3, 1]], [1,0], 1, 1, None, 4],
            [[np.float16, 0, [4, 192, 75, 1]], [np.float16, 0, [384, 48, 3, 1]], [2,0], 1, 1, None, 4],
            [[np.float16, 0, [4, 128, 75, 1]], [np.float16, 0, [128, 32, 3, 1]], [2,0], 1, 1, None, 4],
            [[np.float16, 0, [4, 128, 75, 5]], [np.float16, 0, [128, 32, 3, 3]], [2,1], 1, 1, None, 4],
            [[np.float16, 3, [4, 192, 75, 1]], [np.float16, 0, [384, 48, 3, 1]], [2,0], 1, 1, None, 4],
            [[np.float16, 3, [4, 128, 75, 1]], [np.float16, 0, [128, 32, 3, 1]], [2,0], 1, 1, None, 4],
            [[np.float16, 3, [4, 128, 75, 5]], [np.float16, 0, [128, 32, 3, 3]], [2,1], 1, 1, None, 4],
            [[np.float16, 3, [4, 192, 75, 1]], [np.float16, 4, [384, 48, 3, 1]], [2,0], 1, 1, None, 4],
            [[np.float16, 3, [4, 128, 75, 1]], [np.float16, 4, [128, 32, 3, 1]], [2,0], 1, 1, None, 4],
            [[np.float16, 3, [4, 128, 75, 5]], [np.float16, 4, [128, 32, 3, 3]], [2,1], 1, 1, None, 4],
            [[np.float16, 0, [4, 64, 75, 5]], [np.float16, 0, [64, 1, 3, 3]], [2,1], 1, 1, None, 64], 
            [[np.float16, 0, [4, 64, 75, 1]], [np.float16, 0, [64, 1, 3, 1]], 0, 1, 1, None, 64], 
            [[np.float16, 0, [4, 64, 1, 75]], [np.float16, 0, [64, 1, 1, 3]], 0, 1, 1, None, 64], 
            # 当前不支持kernel_size_h >= padding_h*2 + input_h和kernel_size_w >= padding_w*2 + input_w, 预计330支持
            # [[np.float16, 0, [4, 64, 75, 1]], [np.float16, 0, [128, 16, 3, 3]], 0, 1, 1, None, 4],
            # [[np.float16, 0, [4, 64, 75, 1]], [np.float16, 0, [128, 16, 3, 3]], [1,1], 1, 1, None, 4],
            # [[np.float16, 0, [4, 64, 1, 75]], [np.float16, 0, [128, 16, 3, 3]], 0, 1, 1, None, 4],
            # [[np.float16, 0, [4, 64, 1, 75]], [np.float16, 0, [128, 16, 3, 3]], [1,1], 1, 1, None, 4],
            # 当前不支持in_channel == groups != out_channel
            # [[np.float32, 0, [4, 64, 75, 5]], [np.float32, 0, [128, 1, 3, 3]], [2,1], 1, 1, None, 64], 
            # [[np.float32, 0, [4, 64, 75, 1]], [np.float32, 0, [128, 1, 3, 1]], 0, 1, 1, None, 64], 
            # [[np.float32, 0, [4, 64, 1, 75]], [np.float32, 0, [128, 1, 1, 3]], 0, 1, 1, None, 64], 
        ]

    def test_group_conv2d_backward_shape_format_fp32(self, device):
        shape_format = [  # input, weight, padding, stride, dilation, bias, groups
            # KDXF
            [[np.float32, 0, [4, 64, 75, 10]], [np.float32, 0, [128, 16, 3, 3]], [2,1], 1, 1, None, 4],
            [[np.float32, 0, [4, 128, 75, 10]], [np.float32, 0, [64, 32, 1, 1]], 0, 1, 1, None, 4],
            [[np.float32, 0, [4, 128, 75, 5]], [np.float32, 0, [256, 32, 3, 3]], [2,1], 1, 1, None, 4],
            [[np.float32, 0, [4, 256, 75, 1]], [np.float32, 0, [384, 64, 3, 1]], [1,0], 1, 1, None, 4],
            [[np.float32, 0, [4, 192, 75, 1]], [np.float32, 0, [384, 48, 3, 1]], [2,0], 1, 1, None, 4],
            [[np.float32, 0, [4, 128, 75, 1]], [np.float32, 0, [128, 32, 3, 1]], [2,0], 1, 1, None, 4],
            [[np.float32, 0, [4, 128, 75, 5]], [np.float32, 0, [128, 32, 3, 3]], [2,1], 1, 1, None, 4],
            [[np.float32, 3, [4, 192, 75, 1]], [np.float32, 0, [384, 48, 3, 1]], [2,0], 1, 1, None, 4],
            [[np.float32, 3, [4, 128, 75, 1]], [np.float32, 0, [128, 32, 3, 1]], [2,0], 1, 1, None, 4],
            [[np.float32, 3, [4, 128, 75, 5]], [np.float32, 0, [128, 32, 3, 3]], [2,1], 1, 1, None, 4],
            [[np.float32, 3, [4, 192, 75, 1]], [np.float32, 4, [384, 48, 3, 1]], [2,0], 1, 1, None, 4],
            [[np.float32, 3, [4, 128, 75, 1]], [np.float32, 4, [128, 32, 3, 1]], [2,0], 1, 1, None, 4],
            [[np.float32, 3, [4, 128, 75, 5]], [np.float32, 4, [128, 32, 3, 3]], [2,1], 1, 1, None, 4],
            [[np.float32, 0, [4, 64, 75, 5]], [np.float32, 0, [64, 1, 3, 3]], [2,1], 1, 1, None, 64], 
            [[np.float32, 0, [4, 64, 75, 1]], [np.float32, 0, [64, 1, 3, 1]], 0, 1, 1, None, 64], 
            [[np.float32, 0, [4, 64, 1, 75]], [np.float32, 0, [64, 1, 1, 3]], 0, 1, 1, None, 64], 
            # 当前不支持kernel_size_h >= padding_h*2 + input_h和kernel_size_w >= padding_w*2 + input_w
            # [[np.float32, 0, [4, 64, 75, 1]], [np.float32, 0, [128, 16, 3, 3]], 0, 1, 1, None, 4],
            # [[np.float32, 0, [4, 64, 75, 1]], [np.float32, 0, [128, 16, 3, 3]], [1,1], 1, 1, None, 4],
            # [[np.float32, 0, [4, 64, 1, 75]], [np.float32, 0, [128, 16, 3, 3]], 0, 1, 1, None, 4],
            # [[np.float32, 0, [4, 64, 1, 75]], [np.float32, 0, [128, 16, 3, 3]], [1,1], 1, 1, None, 4],
            # 当前不支持in_channel == groups != out_channel
            # [[np.float32, 0, [4, 64, 75, 5]], [np.float32, 0, [128, 1, 3, 3]], [2,1], 1, 1, None, 64], 
            # [[np.float32, 0, [4, 64, 75, 1]], [np.float32, 0, [128, 1, 3, 1]], 0, 1, 1, None, 64], 
            # [[np.float32, 0, [4, 64, 1, 75]], [np.float32, 0, [128, 1, 1, 3]], 0, 1, 1, None, 64], 
        ]


instantiate_device_type_tests(TestConv2d, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
