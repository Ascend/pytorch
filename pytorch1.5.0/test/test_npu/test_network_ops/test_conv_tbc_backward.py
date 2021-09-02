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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestConvTbcBackward(TestCase):
    weight_grad = []
    input_grad = []

    def getWeightGrad(self, grad):
        self.weight_grad.append(grad.to("cpu"))

    def getInputGrad(self, grad):
        self.input_grad.append(grad.to("cpu"))

    def cpu_op_exec(self, input1, weight1, bias1, pad):
        input1.requires_grad = True
        input1.register_hook(lambda grad: self.getInputGrad(grad))
        weight1.requires_grad = True
        weight1.register_hook(lambda grad: self.getWeightGrad(grad))
        bias1.requires_grad = True
        cpuOutput = torch.conv_tbc(input1, weight1, bias1, pad)
        tmp = torch.ones_like(cpuOutput)
        cpuOutput.backward(tmp)
        cpuOutput = cpuOutput.detach().numpy()
        return cpuOutput, bias1.grad

    def npu_op_exec(self, input1, weight1, bias1, pad):
        input1.requires_grad = True
        input1.register_hook(lambda grad: self.getInputGrad(grad))
        weight1.requires_grad = True
        weight1.register_hook(lambda grad: self.getWeightGrad(grad))
        bias1.requires_grad = True
        npuOutput = torch.conv_tbc(input1, weight1, bias1, pad)
        tmp = torch.ones_like(npuOutput)
        tmp = tmp.to("npu")
        npuOutput.backward(tmp)
        npuOutput = npuOutput.to("cpu")
        npuOutput = npuOutput.detach().numpy()
        return npuOutput, bias1.grad.to("cpu")

    def test_conv_tbc_backward_shape_format(self, device):  

        shape_format = [  # input(TBC1), weight(Lc1c0), bias(c0), pad
            [[np.float16, -1, (5, 1, 2)], [np.float16, -1, (1, 2, 2)], [np.float16, -1, (2)], 0],
            [[np.float32, -1, (5, 2, 2)], [np.float32, -1, (2, 2, 2)], [np.float32, -1, (2)], 1],
            [[np.float16, -1, (256, 8, 1)], [np.float16, -1, (10, 1, 1)], [np.float16, -1, (1)], 0],
            [[np.float16, -1, [232, 23, 7]], [np.float16, -1, [23, 7, 8]], [np.float16, -1, [8]], 1],
            [[np.float32, -1, [10, 2, 4]], [np.float32, -1, [2, 4, 2]], [np.float32, -1, [2]], 1],
            [[np.float16, -1, [150, 1, 20]], [np.float16, -1, [35, 20, 4]], [np.float16, -1, [4]], 1],
            [[np.float16, -1, [10, 2, 2]], [np.float16, -1, [3, 2, 3]], [np.float16, -1, [3]], 0],
        ]

        for item in shape_format:
            self.input_grad.clear()
            self.weight_grad.clear()
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 10)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_weight, npu_weight = create_common_tensor(item[1], 0, 10)
            if cpu_weight.dtype == torch.float16:
                cpu_weight = cpu_weight.to(torch.float32)
            cpu_bias, npu_bias = create_common_tensor(item[2], 0, 10)
            if cpu_bias.dtype == torch.float16:
                cpu_bias = cpu_bias.to(torch.float32)
            cpu_output, cpu_dBias = self.cpu_op_exec(cpu_input1, cpu_weight, cpu_bias, item[3])
            npu_output, npu_dBias = self.npu_op_exec(npu_input1, npu_weight, npu_bias, item[3])
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.input_grad[0] = self.input_grad[0].to(self.input_grad[1].dtype)
            self.weight_grad[0] = self.weight_grad[0].to(self.weight_grad[1].dtype)
            cpu_dBias = cpu_dBias.to(npu_dBias.dtype)
            self.assertRtolEqual(cpu_output, npu_output, 1e-2)
            self.assertRtolEqual(cpu_dBias, npu_dBias)
            self.assertRtolEqual(self.input_grad[0].numpy(), self.input_grad[1].numpy(), 1e-1)
            self.assertRtolEqual(self.weight_grad[0].numpy(), self.weight_grad[1].numpy(), 1e-1)

instantiate_device_type_tests(TestConvTbcBackward, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
