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


class TestConv2dBackward(TestCase):
    weight_grad = []
    input_grad = []

    def getWeightGrad(self, grad):
        self.weight_grad.append(grad.to("cpu"))

    def getInputGrad(self, grad):
        self.input_grad.append(grad.to("cpu"))

    def cpu_op_exec(self, input1, weight, padding = 0, stride = 1, bias1 = None):
        input1.requires_grad = True
        input1.register_hook(lambda grad: self.getInputGrad(grad))
        weight.requires_grad = True
        weight.register_hook(lambda grad: self.getWeightGrad(grad))
        bias1.requires_grad = True

        res_forward = nn.functional.conv2d(input1, weight, bias1, stride, padding)
        grads = torch.ones_like(res_forward).float()
        res_forward.backward(grads, retain_graph=True)
        res_forward = res_forward.detach().numpy()
        return res_forward

    def npu_op_exec(self, input1, weight, padding = 0, stride = 1, bias1 = None):
        input1.requires_grad = True
        input1.register_hook(lambda grad: self.getInputGrad(grad))
        weight.requires_grad = True
        weight.register_hook(lambda grad: self.getWeightGrad(grad))
        bias1 = bias1.to("npu")
        bias1.requires_grad = True

        res_forward = nn.functional.conv2d(input1, weight, bias1, stride, padding)
        grads = torch.ones_like(res_forward).float()
        grads = grads.to("npu")
        res_forward.backward(grads, retain_graph=True)
        res_forward = res_forward.to("cpu")
        res_forward = res_forward.detach().numpy()
        return res_forward

    def test_conv2d_backward_shape_format(self, device):
        shape_format = [  # input, weight, padding, stride
            [[np.float32, 0, (1, 4, 5, 5)], [np.float32, 0, (4, 4, 3, 3)], 0, (1, 1)],
            [[np.float32, 0, (1, 8, 3, 3)], [np.float32, 0, (8, 8, 1, 1)], 0, (2, 1)],
            [[np.float32, 0, (1024, 2048, 6, 6)], [np.float32, 0, (2048, 2048, 3, 3)], 0, (1, 2)],
            [[np.float32, 0, (512, 256, 4, 4)], [np.float32, 0, (256, 256, 2, 2)], 0, (2, 2)],
            [[np.float32, 0, (128, 4, 3, 3)], [np.float32, 0, (4, 4, 2, 2)], 0, (3, 1)],
            [[np.float32, 0, (2, 64, 3, 3)], [np.float32, 0, (64, 64, 3, 3)], 0, (1, 3)],
            [[np.float32, 0, (64, 2, 8, 8)], [np.float32, 0, (2, 2, 1, 1)], 0, (3, 3)],
            [[np.float32, 0, (32, 16, 4, 4)], [np.float32, 0, (16, 16, 3, 3)], 0, (2, 1)],
            [[np.float32, 0, (1024, 8, 3, 3)], [np.float32, 0, (8, 8, 1, 1)], 0, (1, 2)],
            [[np.float32, 0, (1, 8, 512, 512)], [np.float32, 0, (8, 8, 3, 3)], 0, (2, 2)],
            [[np.float32, 0, (1, 2, 1, 1)], [np.float32, 0, (1, 1, 2, 2)], 0, (1, 1)],
        ]

        for item in shape_format:
            self.weight_grad.clear()
            self.input_grad.clear() 
            cpu_input1, npu_input1 = create_common_tensor(item[0], -2, 2)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -2, 2)
            cpu_bias = torch.randn(item[1][2][0])
            npu_bias = copy.deepcopy(cpu_bias)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, item[2], item[3], cpu_bias)
            npu_output = self.npu_op_exec(npu_input1, npu_input2, item[2], item[3], npu_bias)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(self.input_grad[0], self.input_grad[1])
            self.assertRtolEqual(self.weight_grad[0], self.weight_grad[1])


instantiate_device_type_tests(TestConv2dBackward, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
