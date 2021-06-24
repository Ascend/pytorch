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

class TestSlowConvTranspose3d(TestCase):
    def cpu_op_exec(self, input_x, in_channels, out_channels, kernel_size):
        m = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        output = m(input_x)
        return output.detach().numpy()
 
    def cpu_op_exec_fp16(self, input_x, in_channels, out_channels, kernel_size):
        input_x = input_x.to(torch.float32)
        m = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)  
        output = m(input_x)
        return output.detach().numpy()

    def npu_op_exec(self, input_x, in_channels, out_channels, kernel_size):
        m = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        output = m(input_x)
        output = output.to("cpu")
        return output.detach().numpy()
    
    def test_slow_conv_transpose3d(self, device):
        
        shape_format = [ 
            [[np.float16, -1, [20, 16, 10, 50, 100]], 16, 33, 3],
            [[np.float32, -1, [20, 16, 10, 50, 100]], 16, 33, 3],
            [[np.float16, -1, [6, 12, 12, 60, 120]], 12, 25, 3],
            [[np.float32, -1, [10, 8, 6, 30, 60]], 8, 17, 2],
        ]
        for item in shape_format:
            input_x_cpu, input_x_npu = create_common_tensor(item[0], 0, 1)
            if input_x_cpu.dtype == torch.float16:
                cpu_output = self.cpu_op_exec_fp16(input_x_cpu, item[1], item[2], item[3])
            else:
                cpu_output = self.cpu_op_exec(input_x_cpu, item[1], item[2], item[3])

instantiate_device_type_tests(TestSlowConvTranspose3d, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:5")
    run_tests()