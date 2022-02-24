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
import numpy as np

from torch_npu.testing.common_utils import TestCase, run_tests
from torch_npu.testing.common_device_type import instantiate_device_type_tests
from torch_npu.testing.util_test import create_common_tensor

class TestRoll(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input_x = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input = torch.from_numpy(input_x)
        return npu_input

    def cpu_op_exec(self, input_x, shifts, dims):
        output = torch.roll(input_x, shifts, dims).numpy()
        return output

    def npu_op_exec(self, input_x, shifts, dims):
        input1 = input_x.to("npu")
        output = torch.roll(input1, shifts, dims)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_roll_3_4_5_float32(self, device):
        input_x1 = self.generate_data(-1, 1, (3, 4, 5), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, [2, 1], [0, 1])
        npu_output1 = self.npu_op_exec(input_x1, [2, 1], [0, 1])
        self.assertRtolEqual(cpu_output1, npu_output1)
    
    def test_roll_3_4_5_float16(self, device):
        input_x1 = self.generate_data(-1, 1, (3, 4, 5), np.float16)
        input_cpu = input_x1.float()
        cpu_output1 = self.cpu_op_exec(input_cpu, [2, 1], [0, 1]).astype(np.float16)
        npu_output1 = self.npu_op_exec(input_x1, [2, 1], [0, 1])
        self.assertRtolEqual(cpu_output1, npu_output1)
    
    def test_roll_30_40_50_int32(self, device):
        input_x1 = self.generate_data(-1, 1, (30, 40, 50), np.int32)
        cpu_output1 = self.cpu_op_exec(input_x1, [20], [])
        npu_output1 = self.npu_op_exec(input_x1, [20], [])
        self.assertRtolEqual(cpu_output1, npu_output1)
    
    def test_roll_20_30_40_50_uint8(self, device):
        input_x1 = self.generate_data(-1, 1, (20, 30, 40, 50), np.uint8)
        cpu_output1 = self.cpu_op_exec(input_x1, [-20, 30], [-1, 0])
        npu_output1 = self.npu_op_exec(input_x1, [-20, 30], [-1, 0])
        self.assertRtolEqual(cpu_output1, npu_output1)
    
    def test_roll_20_30_40_50_flaot32(self, device):
        input_x1 = self.generate_data(-1, 1, (20, 30, 40, 50), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, [30], [3])
        npu_output1 = self.npu_op_exec(input_x1, [30], [3])
        self.assertRtolEqual(cpu_output1, npu_output1)
   
instantiate_device_type_tests(TestRoll, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()