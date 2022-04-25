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
from graph_utils import graph_mode

class TestSlowConvTranspose2dBackward(TestCase):
    weight_grad = []
    input_grad = []
    bias_grad = []

    def getWeightGrad(self, grad):
        self.weight_grad.append(grad.to("cpu"))

    def getInputGrad(self, grad):
        self.input_grad.append(grad.to("cpu"))
    
    def getBiasGrad(self, grad):
        self.bias_grad.append(grad.to("cpu"))

    def cpu_op_exec(self, input1, weight, bias):
        input1.requires_grad = True
        input1.register_hook(lambda grad: self.getInputGrad(grad))
        weight.requires_grad = True
        weight.register_hook(lambda grad: self.getWeightGrad(grad))
        bias.requires_grad = True
        bias.register_hook(lambda grad: self.getBiasGrad(grad))

        res_forward = torch.nn.functional.conv_transpose2d(input1, weight, bias)
        grads = torch.ones_like(res_forward).float()
        res_forward.backward(grads)
        res_forward = res_forward.detach().numpy()
        return res_forward


    def npu_op_exec(self, input1, weight, bias):
        input1.requires_grad = True
        input1.register_hook(lambda grad: self.getInputGrad(grad))
        weight.requires_grad = True
        weight.register_hook(lambda grad: self.getWeightGrad(grad))
        bias.requires_grad = True
        bias.register_hook(lambda grad: self.getBiasGrad(grad))
        res_forward = torch.nn.functional.conv_transpose2d(input1, weight, bias)
        grads = torch.ones_like(res_forward).float()
        grads = grads.to("npu")
        res_forward.backward(grads)
        res_forward = res_forward.to("cpu")
        res_forward = res_forward.detach().numpy()
        return res_forward

    @graph_mode
    def test_slow_conv_transpose2d_backward_shape_format(self, device):
        shape_format = [ 
                [[np.float32, 0, (1, 4, 5, 5)], [np.float32, 0, (4, 4, 3, 3)], [np.float32, 0, 4]]
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -2, 2)
            cpu_weight, npu_weight = create_common_tensor(item[1], -2, 2)
            cpu_bias, npu_bias = create_common_tensor(item[2], -2, 2)

            cpu_output = self.cpu_op_exec(cpu_input1, cpu_weight, cpu_bias)
            npu_output = self.npu_op_exec(npu_input1, npu_weight, npu_bias)
            # fp32精度不足，放宽对其精度要求
            self.assertRtolEqual(cpu_output, npu_output, prec=1.e-1)
            self.assertRtolEqual(self.input_grad[0], self.input_grad[1], prec=1.e-1)
            self.assertRtolEqual(self.weight_grad[0], self.weight_grad[1], prec=1.e-1)
            self.assertRtolEqual(self.bias_grad[0], self.bias_grad[1], prec=1.e-1)

instantiate_device_type_tests(TestSlowConvTranspose2dBackward, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
