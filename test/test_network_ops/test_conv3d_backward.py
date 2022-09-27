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

import sys
import torch
import torch_npu
import numpy as np
import torch.nn as nn
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor

class TestConv3dBackward(TestCase):
    weight_grad = []
    input_grad = []

    def getWeightGrad(self, grad):
        self.weight_grad.append(grad.to("cpu"))

    def getInputGrad(self, grad):
        self.input_grad.append(grad.to("cpu"))

    def op_exec(self, npu_flag, input1, weight, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1,
                bias=False, groups=1):
        weight1 = weight
        input1.requires_grad = True
        input1.register_hook(lambda grad: self.getInputGrad(grad))

        m1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias, groups=groups)
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

    def test_conv3d_backward_shape_format_fp32(self, device='npu'):
        shape_format = [  # input, weight, padding, stride, dilation, bias, groups
            [[np.float32, 30, [128, 128, 4, 14, 14]],
             [np.float32, 30, [128, 128, 3, 3, 3]], [1,1,1], [1,1,1], 1, None, 1],
            [[np.float32, 30, [128, 64, 4, 14, 14]],
             [np.float32, 30, [128, 64, 3, 3, 3]], [1,1,1], [2,2,2], 1, None, 1],
            [[np.float32, 30, [128, 256, 2, 7, 7]],
             [np.float32, 30, [256, 256, 3, 3, 3]], [1,1,1], [1,1,1], 1, None, 1],
            [[np.float32, 30, [128, 256, 2, 7, 7]],
             [np.float32, 30, [512, 256, 1, 1, 1]], 0, [2,2,2], 1, None, 1]
        ]
        for item in shape_format:
            self.weight_grad.clear()
            self.input_grad.clear()
            input_cpu, input_npu = create_common_tensor(item[0], 0, 1)
            if input_cpu.dtype == torch.float16:
                input_cpu = input_cpu.to(torch.float32)
            weight_cpu, weight_npu = create_common_tensor(item[1], 0, 1)
            if weight_cpu.dtype == torch.float16:
                weight_cpu = weight_cpu.to(torch.float32)
            kernel_size = (item[1][2][2], item[1][2][3],item[1][2][4])
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
            self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.cpu().detach().numpy())
            self.assertRtolEqual(self.input_grad[0].numpy(), self.input_grad[1].cpu().numpy())
            self.assertRtolEqual(self.weight_grad[0].numpy(), self.weight_grad[1].cpu().numpy())


if __name__ == "__main__":
    run_tests()
