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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestDimArange(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input_x = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input = torch.from_numpy(input_x)
        return npu_input

    def cpu_op_exec(self, input_x, dim):
        output = torch._dim_arange(input_x, dim)
        output = output.numpy().astype(np.int32)
        return output

    def npu_op_exec(self, input_x, dim):
        input1 = input_x.to("npu")
        output = torch._dim_arange(input1, dim)
        output = output.to("cpu")
        output = output.numpy()
        return output
        
    def test_dim_arange_3_4_5_0_float32(self, device):
        input_x1 = self.generate_data(-1, 1, (3, 4, 5), np.float32)
        cpu_output = self.cpu_op_exec(input_x1, 1)
        npu_output = self.npu_op_exec(input_x1, 1)
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test_dim_arange_30_40_50_0_float32(self, device):
        input_x1 = self.generate_data(-1, 1, (30, 40, 50), np.float32)
        cpu_output = self.cpu_op_exec(input_x1, 0)
        npu_output = self.npu_op_exec(input_x1, 0)
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test_dim_arange_10_10_10_10_10_10_10_2_float32(self, device):
        input_x1 = self.generate_data(-1, 1, (10, 10, 10, 10, 10, 10), np.float32)
        cpu_output = self.cpu_op_exec(input_x1, 2)
        npu_output = self.npu_op_exec(input_x1, 2)
        self.assertRtolEqual(cpu_output, npu_output)
        
    def test_dim_arange_7_13_22_193_45_2_float16(self, device):
        input_x1 = self.generate_data(-1, 1, (7, 13, 22, 193, 45, 2), np.float16)
        cpu_output = self.cpu_op_exec(input_x1, 2)
        npu_output = self.npu_op_exec(input_x1, 2)
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test_dim_arange_7_13_22_float16(self, device):
        input_x1 = self.generate_data(-1, 1, (7, 13, 22), np.float16)
        cpu_output = self.cpu_op_exec(input_x1, 0)
        npu_output = self.npu_op_exec(input_x1, 0)
        self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestDimArange, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
