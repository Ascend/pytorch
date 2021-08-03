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
import numpy as np
import torch.nn as nn
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestConvolutionDoubleBackward(TestCase):
    def op_exec(self, npu_flag, input, weight, in_channels, out_channels, kernel_size, 
                    padding=0, stride=1, dilation=1, bias=True, groups=1):
        input1 = input
        weight1 = weight
        input1.requires_grad = True

        m1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias, groups=groups)
        m1.weight.data = weight1
        if npu_flag:
            m1 = m1.to("npu")
        output = m1(input1)
        grads = torch.autograd.grad(outputs=output, inputs=(input1, m1.weight), grad_outputs=torch.ones_like(output),
                          retain_graph=True, create_graph=True, only_inputs=True)
        input_grads, weight_grads = grads
        input_grads.retain_grad()
        weight_grads.retain_grad()
        loss = torch.sum(input_grads ** 2) + torch.sum(weight_grads ** 2)
        loss.backward(torch.ones_like(loss))
        input_grads_grad = input_grads.grad
        weight_grads_grad = weight_grads.grad
        if npu_flag:
            output = output.to("cpu")
            input_grads_grad = input_grads_grad.to("cpu")
            weight_grads_grad = weight_grads_grad.to("cpu")
        return output, input_grads_grad, weight_grads_grad

    def test_convolution_double_backward_shape_format_fp16(self, device):
        shape_format = [           
            [[np.float16, 3, [1024, 232, 7, 7]], [np.float16, 4, [232, 232, 1, 1]], 0, 1, 1, None, 1],
            [[np.float16, 0, [1024, 116, 14, 14]], [np.float16, 4, [116, 116, 1, 1]], 0, 1, 1, None, 1],
            [[np.float16, 0, [4, 384, 1, 75]], [np.float16, 0, [192, 384, 1, 3]], 0, 1, 1, None, 1], 
            [[np.float16, 3, [4, 256, 75, 5]], [np.float16, 4, [128, 256, 3, 3]], [2,1], 1, 1, None, 1], 
            [[np.float16, 3, [4, 384, 75, 1]], [np.float16, 4, [192, 384, 3, 1]], 0, 1, 1, None, 1],  
            [[np.float16, 0, [4, 384, 75, 1]], [np.float16, 4, [192, 384, 3, 1]], 0, 1, 1, None, 1], 
            [[np.float16, 0, [4, 384, 1, 75]], [np.float16, 4, [192, 384, 1, 3]], 0, 1, 1, None, 1]
        ]
        for item in shape_format:
            input_cpu, input_npu = create_common_tensor(item[0], -1, 1)
            if input_cpu.dtype == torch.float16:
                input_cpu = input_cpu.to(torch.float32)
            weight_cpu, weight_npu = create_common_tensor(item[1], -1, 1)
            if weight_cpu.dtype == torch.float16:
                weight_cpu = weight_cpu.to(torch.float32)
            kernel_size = (item[1][2][2], item[1][2][3])
            assert item[0][2][1]/item[6] == item[1][2][1]
            cpu_output, cpu_input_grads_grad, cpu_weight_grads_grad = self.op_exec(0, input_cpu, weight_cpu,
                                item[0][2][1], item[1][2][0], kernel_size=kernel_size,padding=item[2],
                                stride=item[3], dilation=item[4], bias=item[5], groups=item[6])
            weight_npu = weight_npu.to("cpu")
            npu_output, npu_input_grads_grad, npu_weight_grads_grad = self.op_exec(1, input_npu, weight_npu,
                                item[0][2][1], item[1][2][0], kernel_size=kernel_size, padding=item[2],
                                stride=item[3], dilation=item[4], bias=item[5], groups=item[6])

            npu_output = npu_output.to(torch.float16)
            npu_input_grads_grad = npu_input_grads_grad.to(torch.float16)
            npu_weight_grads_grad = npu_weight_grads_grad.to(torch.float16)
            cpu_output = cpu_output.to(torch.float16)
            cpu_input_grads_grad = cpu_input_grads_grad.to(torch.float16)
            cpu_weight_grads_grad = cpu_weight_grads_grad.to(torch.float16)

            self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().numpy())
            self.assertRtolEqual(cpu_input_grads_grad.numpy(), npu_input_grads_grad.numpy())
            self.assertRtolEqual(cpu_weight_grads_grad.numpy(), npu_weight_grads_grad.numpy())

instantiate_device_type_tests(TestConvolutionDoubleBackward, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
