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
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestConv1d(TestCase):
    def op_exec_cpu(self, input, weight, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1, bias=True):
        m = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias, groups=1)
        m.weight.data = weight
        output = m(input)
        output = output.detach().numpy()
        return output

    def op_exec_npu(self, input, weight, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1, bias=True):
        m = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias, groups=1)
        m.weight.data = weight
        m = m.to("npu")
        output = m(input)
        output = output.to("cpu")
        output = output.detach().numpy()
        return output

    def test_conv1d_shape_format(self, device):
        shape_format = [  
            [[np.float32, 3, (256, 32, 1, 1)], [np.float32, 3, (8, 32, 1, 1)], 0, (1, 1), (1, 1), (8)],
            [[np.float32, 3, [256, 32, 112, 112]], [np.float32, 0, [16, 32, 1, 1]], 0, 1, 1, None],
            [[np.float32, 0, [256, 3, 224, 224]], [np.float32, 0, [32, 3, 3, 3]], 0, [2, 2], 1, None],
            [[np.float32, 3, (2, 3, 3, 3)], [np.float32, 0, (3, 1, 3, 3)], 3, 1, 1, 1],
            [[np.float32, 3, [1024, 232, 7, 7]], [np.float32, 4, [232, 232, 1, 1]], 0, 1, 1, True],
        ]

        for item in shape_format:
            input_cpu, input_npu = create_common_tensor(item[0], -2, 2)
            weight_cpu, weight_npu = create_common_tensor(item[1], -2, 2)
            kernel_size = (item[1][2][2], item[1][2][3])
            cpu_output = self.op_exec_cpu(input_cpu, weight_cpu, item[0][2][1], item[1][2][0], kernel_size=kernel_size,
                                          padding=item[2], stride=item[3], dilation=item[4], bias=item[5])
            weight_npu = weight_npu.to("cpu")
            npu_output = self.op_exec_npu(input_npu, weight_npu, item[0][2][1], item[1][2][0], kernel_size=kernel_size,
                                          padding=item[2], stride=item[3], dilation=item[4], bias=item[5])            
            self.assertRtolEqual(cpu_output, npu_output)
            
    def test_conv1d_shape_format_float16(self, device):
        def cpu_op_exec_fp16(input, weight, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1, bias=True):
            input = input.to(torch.float32)
            weight = weight.to(torch.float32)
            m = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias, groups=1)
            m.weight.data = weight
            output = m(input)
            output = output.detach().numpy()
            output = output.astype(np.float16)
            return output

        shape_format = [  
            [[np.float16, 3, (256, 32, 1, 1)], [np.float16, 3, (8, 32, 1, 1)], 0, (1, 1), (1, 1), (8)],
            [[np.float16, 3, [256, 32, 112, 112]], [np.float16, 0, [16, 32, 1, 1]], 0, 1, 1, None],
            [[np.float16, 0, [256, 3, 224, 224]], [np.float16, 0, [32, 3, 3, 3]], 0, [2, 2], 1, None],
            [[np.float16, 3, (2, 3, 3, 3)], [np.float16, 0, (3, 1, 3, 3)], 3, 1, 1, 1],
            [[np.float16, 3, [1024, 232, 7, 7]], [np.float16, 4, [232, 232, 1, 1]], 0, 1, 1, True],
        ]

        for item in shape_format:
            input_cpu, input_npu = create_common_tensor(item[0], -2, 2)
            weight_cpu, weight_npu = create_common_tensor(item[1], -2, 2)
            kernel_size = (item[1][2][2], item[1][2][3])
            cpu_output = cpu_op_exec_fp16(input_cpu, weight_cpu, item[0][2][1], item[1][2][0], kernel_size=kernel_size,
                                          padding=item[2], stride=item[3], dilation=item[4], bias=item[5])
            weight_npu = weight_npu.to("cpu")
            npu_output = self.op_exec_npu(input_npu, weight_npu, item[0][2][1], item[1][2][0], kernel_size=kernel_size,
                                          padding=item[2], stride=item[3], dilation=item[4], bias=item[5])            
            self.assertRtolEqual(cpu_output, npu_output)          

instantiate_device_type_tests(TestConv1d, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:6")
    run_tests()