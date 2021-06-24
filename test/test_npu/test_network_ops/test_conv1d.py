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
import torch.nn.functional as F
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestConv1d(TestCase):
    def cpu_op_exec(self, input1, weight, stride, pad):
        input1.requires_grad = True
        weight.requires_grad = True
        out = F.conv1d(input1, weight, stride=stride, padding=pad)
        out.backward(torch.ones_like(out))
        input_grad = input1.grad
        weight_grad = weight.grad
        out = out.detach()
        return out, input_grad, weight_grad
    
    def npu_op_exec(self, input1, weight, stride, pad):
        input1.requires_grad = True
        weight.requires_grad = True
        out = F.conv1d(input1, weight, stride=stride, padding=pad)
        out.backward(torch.ones_like(out))
        input_grad = input1.grad.cpu()
        weight_grad = weight.grad.cpu()
        out = out.cpu().detach()
        return out, input_grad, weight_grad

    def test_conv1d_shape_format_fp16(self, device):
        shape_format = [
            [[np.float16, 0, [4, 1, 166400]], [np.float16, 0, [514, 1, 400]], 400, 0]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 1)
            cpu_weight, npu_weight = create_common_tensor(item[1], 0, 1)
            stride = item[2]
            padding = item[3]
            
            cpu_output, cpu_input_grad, cpu_weight_grad = self.cpu_op_exec(cpu_input.float(), cpu_weight.float(), stride, padding)
            cpu_output = cpu_output.half()
            cpu_input_grad = cpu_input_grad.half()
            cpu_weight_grad = cpu_weight_grad.half()
            npu_output, npu_input_grad, npu_weight_grad = self.npu_op_exec(npu_input, npu_weight, stride, padding)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_input_grad, npu_input_grad)
            self.assertRtolEqual(cpu_weight_grad, npu_weight_grad)

    def test_conv1d_shape_format_fp32(self, device):
        shape_format = [
            [[np.float32, 0, [4, 1, 166400]], [np.float32, 0, [514, 1, 400]], 400, 0]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 1)
            cpu_weight, npu_weight = create_common_tensor(item[1], 0, 1)
            stride = item[2]
            padding = item[3]
            
            cpu_output, cpu_input_grad, cpu_weight_grad = self.cpu_op_exec(cpu_input, cpu_weight, stride, padding)
            npu_output, npu_input_grad, npu_weight_grad = self.npu_op_exec(npu_input, npu_weight, stride, padding)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_input_grad, npu_input_grad)
            self.assertRtolEqual(cpu_weight_grad, npu_weight_grad)


instantiate_device_type_tests(TestConv1d, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
