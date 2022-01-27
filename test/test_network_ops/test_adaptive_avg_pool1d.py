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
import torch_npu
import torch.nn as nn
import numpy as np

from torch_npu.testing.common_utils import TestCase, run_tests
from torch_npu.testing.common_device_type import Dtypes, instantiate_device_type_tests
from torch_npu.testing.util_test import create_common_tensor, test_2args_broadcast, create_dtype_tensor, UT_FAST_MODE

class TestAdaptiveAvgPool1d(TestCase):
    def cpu_op_exec(self, input1, output_size):
        m = nn.AdaptiveAvgPool1d(output_size)
        output = m(input1)
        return output

    def npu_op_exec(self, input1, output_size):
        m = nn.AdaptiveAvgPool1d(output_size).npu()
        output = m(input1)
        return output.cpu()
    
    def test_AdaptiveAvgPool1d_shape_format_fp16(self, device):
        shape_format = [
                [np.float16, 0, (64, 10, 16)],
                [np.float16, -1, (256, 2048, 8)],
                [np.float16, 3, (32, 16, 16)]
        ]
        output_list = [(4), (3)]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 10)
            for output_size in output_list:
                cpu_output = self.cpu_op_exec(cpu_input.float(), output_size).half()
                npu_output = self.npu_op_exec(npu_input, output_size)
                self.assertRtolEqual(cpu_output, npu_output, prec16=0.002)

    def test_AdaptiveAvgPool1d_shape_format_fp32(self, device):
        shape_format = [
                [np.float32, 0, (64, 10, 16)],
                [np.float32, -1, (256, 2048, 8)],
                [np.float32, 3, (32, 16, 16)]
        ]
        output_list = [(4), (3), (1)]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 10)
            for output_size in output_list:
                cpu_output = self.cpu_op_exec(cpu_input, output_size)
                npu_output = self.npu_op_exec(npu_input, output_size)
                self.assertRtolEqual(cpu_output, npu_output, 0.001)

instantiate_device_type_tests(TestAdaptiveAvgPool1d, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
