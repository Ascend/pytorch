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

class TestVar(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input_x = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input = torch.from_numpy(input_x)
        return npu_input

    def cpu_var_exec(self, input, dim, unbiased=True, keepdim=False):
        cpu_variance = torch.var(input, dim, unbiased, keepdim)
        return cpu_variance.numpy()
    
    def cpu_var_mean_exec(self, input, dim, unbiased=True, keepdim=False):
        cpu_variance, cpu_mean = torch.var_mean(input, dim, unbiased, keepdim)
        return cpu_variance.numpy(), cpu_mean.numpy()
    
    def cpu__var_exec(self, input, unbiased=True):
        cpu_variance = torch._var(input, unbiased)
        return cpu_variance.numpy()

    def npu_var_exec(self, input, dim, unbiased=True, keepdim=False):
        input = input.to("npu")
        npu_variance = torch.var(input, dim, unbiased, keepdim)
        return npu_variance.cpu().numpy()
    
    def npu_var_exec_out(self, input, output_y, dim, unbiased=True, keepdim=False):
        input = input.to("npu")
        output_y = output_y.to("npu")
        torch.var(input, dim, unbiased, keepdim, out=output_y)
        return output_y.cpu().numpy()
    
    def npu_var_mean_exec(self, input, dim, unbiased=True, keepdim=False):
        input = input.to("npu")
        npu_variance, npu_mean = torch.var_mean(input, dim, unbiased, keepdim)
        return npu_variance.cpu().numpy(), npu_mean.cpu().numpy()
    
    def npu__var_exec(self, input, unbiased=True):
        input = input.to("npu")
        npu_variance = torch._var(input, unbiased)
        return npu_variance.cpu().numpy()

    def test_var_fp16(self, device):
        input_x1 = self.generate_data(-1, 1, (30, 40, 50), np.float16)
        cpu_output = self.cpu_var_exec(input_x1, [1], True, False)
        npu_output = self.npu_var_exec(input_x1, [1], True, False)
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test_var_fp16_out(self, device):
        input_x1 = self.generate_data(-1, 1, (30, 40, 50), np.float16)
        output_y = self.generate_data(-1, 1, (30, 50), np.float16)
        cpu_output = self.cpu_var_exec(input_x1, [1], True, False)
        npu_output = self.npu_var_exec_out(input_x1, output_y, [1], True, False)
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test_var_fp16_names_out(self, device):
        input_x1 = self.generate_data(-1, 1, (30, 40, 50), np.float16).rename('a', 'b', 'c')
        output_y = self.generate_data(-1, 1, (30, 50), np.float16)
        cpu_output = self.cpu_var_exec(input_x1, ['b'], True, False)
        npu_output = self.npu_var_exec_out(input_x1, output_y, ['b'], True, False)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_var_fp32_1(self, device):
        input_x1 = self.generate_data(-1, 1, (3, 4, 5, 6), np.float32)
        cpu_output = self.cpu_var_exec(input_x1, [0, 1, 2], True, False)
        npu_output = self.npu_var_exec(input_x1, [0, 1, 2], True, False)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_var_fp16_2(self, device):
        input_x1 = self.generate_data(-1, 1, (30, 40, 13), np.float16)
        input_x1.names = ['A', 'B', 'C']
        cpu_output = self.cpu_var_exec(input_x1, 'B', True, False)
        npu_output = self.npu_var_exec(input_x1, 'B', True, False)
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test_var_fp32_2(self, device):
        input_x1 = self.generate_data(-1, 1, (30, 40, 13), np.float32)
        input_x1.names = ['A', 'B', 'C']
        cpu_output = self.cpu_var_exec(input_x1, 'B', True, False)
        npu_output = self.npu_var_exec(input_x1, 'B', True, False)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_var_fp32(self, device):
        input_x1 = self.generate_data(-1, 1, (3, 4, 5, 6, 7, 8, 9), np.float32)
        cpu_output = self.cpu_var_exec(input_x1, [0, 3, 5], False, False)
        npu_output = self.npu_var_exec(input_x1, [0, 3, 5], False, False)
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test__var_fp32(self, device):
        input_x1 = self.generate_data(-1, 1, (3, 4, 5, 6, 7, 8, 9), np.float32)
        cpu_output = self.cpu__var_exec(input_x1)
        npu_output = self.npu__var_exec(input_x1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_var_mean_fp32_1(self, device):
        input_x1 = self.generate_data(-1, 1, (3, 4, 3, 5, 7, 9), np.float32)
        cpu_output1, cpu_output2 = self.cpu_var_mean_exec(input_x1, [0, 1, 2, 3], False, False)
        npu_output1, npu_output2 = self.npu_var_mean_exec(input_x1, [0, 1, 2, 3], False, False)
        self.assertRtolEqual(cpu_output1, npu_output1)
        self.assertRtolEqual(cpu_output2, npu_output2)

    def test_var_mean_fp32_2(self, device):
        input_x1 = self.generate_data(-1, 1, (10, 20, 30, 40), np.float32)
        cpu_output1, cpu_output2 = self.cpu_var_mean_exec(input_x1, [0, 1, 2, 3], False, False)
        npu_output1, npu_output2 = self.npu_var_mean_exec(input_x1, [0, 1, 2, 3], False, False)
        self.assertRtolEqual(cpu_output1, npu_output1)
        self.assertRtolEqual(cpu_output2, npu_output2)

    def test_var_mean_fp16_1(self, device):
        input_x1 = self.generate_data(-1, 1, (3, 4, 3, 5, 7, 9), np.float16)
        input_x1.names = ['A', 'B', 'C', 'D', 'E', 'F']
        cpu_output1, cpu_output2 = self.cpu_var_mean_exec(input_x1, ['A', 'B', 'D'], False, False)
        npu_output1, npu_output2 = self.npu_var_mean_exec(input_x1, ['A', 'B', 'D'], False, False)
        self.assertRtolEqual(cpu_output1, npu_output1)
        self.assertRtolEqual(cpu_output2, npu_output2)


instantiate_device_type_tests(TestVar, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()