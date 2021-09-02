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

class TestConvTranspose2dBackward(TestCase):
    weight_grad = []
    input_grad = []

    def getWeightGrad(self, grad):
        self.weight_grad.append(grad.to("cpu"))

    def getInputGrad(self, grad):
        self.input_grad.append(grad.to("cpu"))
    
    def cpu_op_exec(self, input1, weight, bias1):
        input1.requires_grad = True
        input1.register_hook(lambda grad: self.getInputGrad(grad))
        weight.requires_grad = True
        weight.register_hook(lambda grad: self.getWeightGrad(grad))
        bias1.requires_grad = True

        res_forward = nn.functional.conv_transpose2d(input1, weight, padding=1, bias=bias1)
        grads = torch.ones_like(res_forward).float()
        res_forward.backward(grads, retain_graph=True)
        res_forward = res_forward.detach().numpy()
        return res_forward
    
    def npu_op_exec(self, input1, weight, bias1):
        input1.requires_grad = True
        input1.register_hook(lambda grad: self.getInputGrad(grad))
        weight.requires_grad = True
        weight.register_hook(lambda grad: self.getWeightGrad(grad))
        bias1 = bias1.to("npu")
        bias1.requires_grad = True

        res_forward = nn.functional.conv_transpose2d(input1, weight, padding=1, bias=bias1)
        grads = torch.ones_like(res_forward).float()
        grads = grads.to("npu")
        res_forward.backward(grads, retain_graph=True)
        res_forward = res_forward.to("cpu")
        res_forward = res_forward.detach().numpy()
        return res_forward

    # conv类算子不支持fp32数据的精度要求
    def test_conv_transpose2d_backward_common_shape_format(self, device):
        shape_format = [
                [[np.float32, 3, (30, 50, 30, 30)], [np.float32, 4, (50, 60, 20, 20)]],
                [[np.float32, 0, (1, 4, 5, 5)],     [np.float32, 4, (4, 4, 3, 3)]],
                [[np.float32, 4, (5, 10, 20, 20)],  [np.float32, 4, (10, 20, 8, 8)]],
                [[np.float32, 3, (1, 4, 5, 5)],     [np.float32, 4, (4, 8, 3, 3)]],
                [[np.float32, 0, (30, 50, 30, 30)], [np.float32, 0, (50, 60, 20, 20)]],
                [[np.float32, 0, (1, 4, 5, 5)],     [np.float32, 0, (4, 4, 3, 3)]],
                [[np.float32, 0, (5, 10, 20, 20)],  [np.float32, 0, (10, 20, 8, 8)]],
                [[np.float32, 0, (1, 4, 5, 5)],     [np.float32, 0, (4, 8, 3, 3)]]
        ]
        for item in shape_format:
            self.weight_grad.clear()
            self.input_grad.clear()
            cpu_bias = torch.randn(item[1][2][1])
            npu_bias = copy.deepcopy(cpu_bias)
            cpu_input1, npu_input1 = create_common_tensor(item[0], -16, 16)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -16, 16)
            kernel_size = (item[1][2][2], item[1][2][3])
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, bias1=cpu_bias)
            npu_output = self.npu_op_exec(npu_input1, npu_input2, bias1=npu_bias)
            # fp32精度不足，放宽对其精度要求
            self.assertRtolEqual(cpu_output, npu_output, prec=1.e-1)
            self.assertRtolEqual(self.input_grad[0], self.input_grad[1], prec=1.e-1)
            self.assertRtolEqual(self.weight_grad[0], self.weight_grad[1], prec=1.e-1)
    
    def test_conv_transpose2d_backward_fp16_shape_format(self, device):
        def cpu_op_exec_fp16(input1, weight, bias1):
            input1 = input1.to(torch.float32)
            weight = weight.to(torch.float32)
            bias1 = bias1.to(torch.float32)
            input1.requires_grad = True
            input1.register_hook(lambda grad: self.getInputGrad(grad))
            weight.requires_grad = True
            weight.register_hook(lambda grad: self.getWeightGrad(grad))
            bias1.requires_grad = True

            res_forward = nn.functional.conv_transpose2d(input1, weight, padding=1, bias=bias1)
            grads = torch.ones_like(res_forward).float()
            res_forward.backward(grads, retain_graph=True)
            res_forward = res_forward.detach().numpy()
            res_forward = res_forward.astype(np.float16)
            return res_forward
        
        shape_format = [
                [[np.float16, 3, (30, 50, 30, 30)], [np.float16, 4, (50, 60, 20, 20)]],
                [[np.float16, 0, (1, 4, 5, 5)],     [np.float16, 4, (4, 4, 3, 3)]],
                [[np.float16, 4, (5, 10, 20, 20)],  [np.float16, 4, (10, 20, 8, 8)]],
                [[np.float16, 3, (1, 4, 5, 5)],     [np.float16, 4, (4, 8, 3, 3)]],
                [[np.float16, 0, (30, 50, 30, 30)], [np.float16, 0, (50, 60, 20, 20)]],
                [[np.float16, 0, (1, 4, 5, 5)],     [np.float16, 0, (4, 4, 3, 3)]],
                [[np.float16, 0, (5, 10, 20, 20)],  [np.float16, 0, (10, 20, 8, 8)]],
                [[np.float16, 0, (1, 4, 5, 5)],     [np.float16, 0, (4, 8, 3, 3)]]
        ]

        for item in shape_format:
            self.weight_grad.clear()
            self.input_grad.clear()
            cpu_bias = torch.randn(item[1][2][1])
            npu_bias = copy.deepcopy(cpu_bias)
            cpu_input1, npu_input1 = create_common_tensor(item[0], -16, 16)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -16, 16)
            kernel_size = (item[1][2][2], item[1][2][3])
            cpu_output = cpu_op_exec_fp16(cpu_input1, cpu_input2, bias1=cpu_bias)
            npu_output = self.npu_op_exec(npu_input1, npu_input2, bias1=npu_bias)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(self.input_grad[0].to(torch.float16), self.input_grad[1])
            self.assertRtolEqual(self.weight_grad[0].to(torch.float16), self.weight_grad[1])

instantiate_device_type_tests(TestConvTranspose2dBackward, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()