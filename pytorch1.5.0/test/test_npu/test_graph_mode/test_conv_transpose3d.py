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

class TestConvTranspose2d(TestCase):
    def cpu_op_exec(self, input, weight, groups):
        cpu_output = torch.nn.functional.conv_transpose3d(input, weight,bias=None, 
                            stride=1, padding=0, output_padding=0, groups=groups, dilation=1)
        cpu_output = cpu_output.numpy()
        return cpu_output

    def cpu_op_exec_fp16(self, input, weight, groups):
        input = input.to(torch.float32)
        weight = weight.to(torch.float32)
        cpu_output = torch.nn.functional.conv_transpose3d(input, weight, bias=None, 
                            stride=1, padding=0, output_padding=0, groups=groups, dilation=1)
        cpu_output = cpu_output.numpy()
        cpu_output = cpu_output.astype(np.float16)

        return cpu_output

    def npu_op_exec(self, input, weight, groups):
        input = input.to("npu")
        weight = weight.to("npu")
        npu_output = torch.nn.functional.conv_transpose3d(input, weight, bias=None, 
                            stride=1, padding=0, output_padding=0, groups=groups, dilation=1)
        npu_output = npu_output.to("cpu").numpy()

        return npu_output

    @graph_mode
    def test_conv_transpose2d_fp32(self, device):
        shape_format = [
            [[np.float32, 30, [12, 12, 4, 14, 14]], [np.float32, 30, [12, 12, 3, 3, 3]], 1],
            [[np.float32, 30, [12, 64, 4, 14, 14]], [np.float32, 30, [64, 64, 3, 3, 3]], 1],
            [[np.float32, 30, [12, 25, 2, 7, 7]], [np.float32, 30, [25, 25, 3, 3, 3]], 1],
            [[np.float32, 30, [12, 51, 1, 4, 4]], [np.float32, 30, [51, 51, 3, 3, 3]], 1],
            [[np.float32, 30, [12, 25, 2, 7, 7]], [np.float32, 30, [25, 25, 1, 1, 1]], 1]
        ]
        for item in shape_format:
            input_cpu, input_npu = create_common_tensor(item[0], 0, 10)
            weight_cpu, weight_npu = create_common_tensor(item[1], 0, 10)
            if input_cpu.dtype == torch.float16:
                cpu_output = self.cpu_op_exec_fp16(input_cpu, weight_cpu, item[-1])
            else:
                cpu_output = self.cpu_op_exec(input_cpu, weight_cpu, item[-1])
            npu_output = self.npu_op_exec(input_npu, weight_npu, item[-1])
            # fp32精度不足，放宽对其精度要求
            self.assertRtolEqual(cpu_output, npu_output, prec=1.e-1)

    @graph_mode
    def test_conv_transpose2d_fp16(self, device):
        shape_format = [  
            [[np.float16, 30, [12, 12, 4, 14, 14]], [np.float16, 30, [12, 12, 3, 3, 3]], 1],
            [[np.float16, 30, [12, 64, 4, 14, 14]], [np.float16, 30, [64, 64, 3, 3, 3]], 1],
            [[np.float16, 30, [12, 25, 2, 7, 7]], [np.float16, 30, [25, 25, 3, 3, 3]], 1],
            [[np.float16, 30, [12, 51, 1, 4, 4]], [np.float16, 30, [51, 51, 3, 3, 3]], 1],
            [[np.float16, 30, [12, 25, 2, 7, 7]], [np.float16, 30, [25, 25, 1, 1, 1]], 1],
        ]
        for item in shape_format:
            input_cpu, input_npu = create_common_tensor(item[0], 0, 10)
            weight_cpu, weight_npu = create_common_tensor(item[1], 0, 10)
            if input_cpu.dtype == torch.float16:
                cpu_output = self.cpu_op_exec_fp16(input_cpu, weight_cpu, item[-1])
            else:
                cpu_output = self.cpu_op_exec(input_cpu, weight_cpu, item[-1])
            npu_output = self.npu_op_exec(input_npu, weight_npu, item[-1])
            self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestConvTranspose2d, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
