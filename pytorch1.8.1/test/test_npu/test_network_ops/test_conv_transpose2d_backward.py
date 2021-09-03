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
        print("===cpu_res_forward===")
        print(res_forward)
        grads = torch.ones_like(res_forward).float()
        res_forward.backward(grads, retain_graph=True)
        print("===cpu_bias===")
        print(bias1)
        print("===cpu_bias_grad===")
        print(bias1.grad)
        return res_forward
    
    def npu_op_exec(self, input1, weight, bias1):
        input1.requires_grad = True
        input1.register_hook(lambda grad: self.getInputGrad(grad))
        weight.requires_grad = True
        weight.register_hook(lambda grad: self.getWeightGrad(grad))
        bias1 = bias1.to("npu")
        bias1.requires_grad = True

        res_forward = nn.functional.conv_transpose2d(input1, weight, padding=1, bias=bias1)
        print("===npu_res_forward===")
        print(res_forward)
        grads = torch.ones_like(res_forward).float()
        grads = grads.to("npu")
        res_forward.backward(grads, retain_graph=True)
        print("===npu_bias===")
        print(bias1)
        print("===npu_bias_grad===")
        print(bias1.grad)
        res_forward = res_forward.to("cpu")
        return res_forward
    
    def conv_transpose2d_backward_result(self, shape_format):
        cpu_bias = torch.randn(4)
        npu_bias = copy.deepcopy(cpu_bias)

        for item in shape_format:
            self.weight_grad.clear()
            self.input_grad.clear()
            cpu_input1, npu_input1 = create_common_tensor(item[0], -2, 2)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -2, 2)
            if cpu_input2.dtype == torch.float16:
                cpu_input2 = cpu_input2.to(torch.float32)
            kernel_size = (item[1][2][2], item[1][2][3])
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, bias1=cpu_bias)
            npu_output = self.npu_op_exec(npu_input1, npu_input2, bias1=npu_bias)

            print("===input_grad===")
            print(self.input_grad)
            print("===weight_grad===")
            print(self.weight_grad)

            cpu_output = cpu_output.to(torch.float16)
            npu_output = npu_output.to(torch.float16)
            self.input_grad[0] = self.input_grad[0].to(torch.float16)
            self.input_grad[1] = self.input_grad[1].to(torch.float16)
            self.weight_grad[0] = self.weight_grad[0].to(self.weight_grad[1].dtype)

            self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().numpy())
            self.assertRtolEqual(self.input_grad[0].numpy(), self.input_grad[1].numpy())
            self.assertRtolEqual(self.weight_grad[0].numpy(), self.weight_grad[1].numpy())

    def test_conv_transpose2d_backward_shape_format_fp16(self, device):
        shape_format = [
            [[np.float16, 0, [1, 4, 5, 5]], [np.float16, 0, [4, 4, 3, 3]]]
        ]
        self.conv_transpose2d_backward_result(shape_format)

    def test_conv_transpose2d_backward_shape_format_fp32(self, device):
        shape_format = [
            [[np.float32, 0, [1, 4, 5, 5]], [np.float32, 0, [4, 4, 3, 3]]]
        ]
        #conv类算子不支持fp32数据的精度要求
        #self.conv_transpose2d_backward_result(shape_format)


instantiate_device_type_tests(TestConvTranspose2dBackward, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
