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


class TestConvTranspose2d(TestCase):
    def cpu_op_exec(self, input, weight, groups):
        cpu_output = torch.nn.functional.conv_transpose2d(input, weight,bias=None, 
                            stride=1, padding=0, output_padding=0, groups=groups, dilation=1)
        cpu_output = cpu_output.numpy()
        return cpu_output

    def cpu_op_exec_fp16(self, input, weight, groups):
        input = input.to(torch.float32)
        weight = weight.to(torch.float32)
        cpu_output = torch.nn.functional.conv_transpose2d(input, weight, bias=None, 
                            stride=1, padding=0, output_padding=0, groups=groups, dilation=1)
        cpu_output = cpu_output.numpy()
        cpu_output = cpu_output.astype(np.float16)

        return cpu_output

    def npu_op_exec(self, input, weight, groups):
        input = input.to("npu")
        weight = weight.to("npu")
        npu_output = torch.nn.functional.conv_transpose2d(input, weight, bias=None, 
                            stride=1, padding=0, output_padding=0, groups=groups, dilation=1)
        npu_output = npu_output.to("cpu").numpy()

        return npu_output

    def test_conv_transpose2d(self, device):
        shape_format = [  
            # input, weight
            [[np.float16, 3, [1024, 116, 14, 14]], [np.float16, 4, [116, 116, 1, 1]], 1],
            [[np.float16, 3, [1024, 58, 28, 28]], [np.float16, 3, [58, 58, 1, 1]], 1],
            [[np.float16, 4, [1024, 3, 224, 224]], [np.float16, 4, [3, 3, 3, 3]], 1],
            [[np.float16, 0, [1024, 116, 14, 14]], [np.float16, 4, [116, 116, 1, 1]], 1],
            [[np.float16, 3, [1024, 232, 7, 7]], [np.float16, 4, [232, 232, 1, 1]], 1],
            [[np.float16, 4, [1024, 58, 28, 28]], [np.float16, 4, [58, 58, 1, 1]], 1],
            [[np.float16, 0, [1024, 24, 56, 56]], [np.float16, 4, [24, 24, 1, 1]], 1],
            [[np.float32, 0, [256, 128, 7, 7]], [np.float32, 4, [128, 128, 3, 3]], 1],
            [[np.float32, 4, [256, 3, 224, 224]], [np.float32, 4, [3, 3, 7, 7]], 1],
            [[np.float32, 3, [2, 3, 3, 3]], [np.float32, 4, [3, 1, 3, 3]], 1],
            [[np.float32, 3, [1024, 232, 7, 7]], [np.float32, 4, [232, 232, 1, 1]], 1],
            [[np.float16, 3, [1024, 116*3, 14, 14]], [np.float16, 4, [116*3, 150//3, 1, 1]], 3],
            [[np.float16, 3, [1024, 58*2, 28, 28]], [np.float16, 3, [58*2, 58//2, 1, 1]], 2],
            [[np.float16, 0, [1, 3*3, 224, 224]], [np.float16, 0, [3*3, 1, 3, 3]], 3],
            [[np.float16, 0, [1024, 116*4, 14, 14]], [np.float16, 4, [116*4, 116//4, 1, 1]], 4],
            [[np.float32, 3, [1024, 116*3, 14, 14]], [np.float32, 4, [116*3, 150//3, 1, 1]], 3],
            [[np.float32, 3, [1024, 58*2, 28, 28]], [np.float32, 3, [58*2, 58//2, 1, 1]], 2],
            [[np.float32, 0, [1, 3*3, 224, 224]], [np.float32, 0, [3*3, 1, 3, 3]], 3],
            [[np.float32, 0, [1024, 116*4, 14, 14]], [np.float32, 4, [116*4, 116//4, 1, 1]], 4],
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


instantiate_device_type_tests(TestConvTranspose2d, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
