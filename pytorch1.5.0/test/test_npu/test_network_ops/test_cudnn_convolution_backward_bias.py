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
import torch.nn as nn
import sys
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestCudnnConvolutionBackwardBias(TestCase):
    def cpu_op_exec(self, input1):
        m = nn.Conv2d(1,8,(2,3),bias=True)
        m = m.to(torch.float32)
        output = m(input1)
        output.backward(torch.ones_like(output), retain_graph = True)
        grad = m.bias.grad
        return grad.detach().numpy()

    def cpu_op_exec_f16(self, input1):
        input1 = input1.to(torch.float32)
        m = nn.Conv2d(1,8,(2,3),bias=True)
        m = m.to(torch.float32)
        output = m(input1)
        output.backward(torch.ones_like(output), retain_graph = True)
        grad = m.bias.grad
        grad = grad.to(torch.float16)
        return grad.detach().numpy()

    def npu_op_exec(self, input1):
        m = nn.Conv2d(1,8,(2,3),bias=True)
        m = m.to("npu")
        m = m.to(torch.float32)
        output = m(input1)
        output = output.to("npu")
        inputback = torch.ones_like(output)
        output.backward(inputback, retain_graph = True)
        output = output.to("cpu")
        grad = m.bias.grad
        grad = grad.to("cpu")
        return grad.detach().numpy()

    def npu_op_exec_f16(self, input1):
        m = nn.Conv2d(1,8,(2,3),bias=True)
        m = m.to("npu")
        input1 = input1.to(torch.float32)
        m = m.to(torch.float32)
        output = m(input1)
        output = output.to("npu")
        inputback = torch.ones_like(output)
        output.backward(inputback, retain_graph = True)
        output = output.to("cpu")
        grad = m.bias.grad
        grad = grad.to(torch.float16)
        grad = grad.to("cpu")
        return grad.detach().numpy()

    def test_cudnn_convolution_backward_bias(self, device):
        shape_format = [
            [[[np.float32, -1, (10,1,30,32)]],
            [[np.float32, -1, (10, 1, 13, 4)]]],
            [[[np.float16, -1, (1, 1, 2, 3)]],
            [[np.float16, -1, (50, 1, 4, 5)]]]
        ]
        for item in shape_format[0]:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

        for item in shape_format[1]:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_output = self.cpu_op_exec_f16(cpu_input1)
            npu_output = self.npu_op_exec_f16(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestCudnnConvolutionBackwardBias, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
