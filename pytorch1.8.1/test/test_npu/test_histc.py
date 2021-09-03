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

class TestHistc(TestCase):
    def generate_single_data(self, min_d, max_d, shape, dtype): 
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype) 
        npu_input = torch.from_numpy(input1) 
         
        return npu_input 
        
    def cpu_op_exec(self, input1, bins=100, min=0, max=0): 
        output = torch.histc(input1, bins=bins, min=min, max=max)
        output = output.numpy() 
        return output 
    
    def npu_op_exec(self, input1, bins=100, min=0, max=0): 
        input1 = input1.to("npu") 
        output = torch.histc(input1, bins=bins, min=min, max=max)
        output = output.to("cpu") 
        output = output.numpy() 
        return output 

    def test_histc_int32_1(self, device):
        npu_input1 = self.generate_single_data(0, 100, (1000,), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1.to(torch.float), bins=50, min=0, max=100)
        npu_output = self.npu_op_exec(npu_input1, bins=50, min=0, max=100)
        cpu_output = cpu_output.astype(np.int32)
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test_histc_int32_2(self, device):
        npu_input1 = self.generate_single_data(0, 100, (20, 30, 2), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1.to(torch.float), bins=50, min=0, max=100)
        npu_output = self.npu_op_exec(npu_input1, bins=50, min=0, max=100)
        cpu_output = cpu_output.astype(np.int32)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_histc_int32_3(self, device):
        npu_input1 = self.generate_single_data(0, 100, (10000,), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1.to(torch.float), bins=50, min=0, max=100)
        npu_output = self.npu_op_exec(npu_input1, bins=50, min=0, max=100)
        cpu_output = cpu_output.astype(np.int32)
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test_histc_int32_4(self, device):
        npu_input1 = self.generate_single_data(0, 100, (10000,), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1.to(torch.float), bins=5000, min=0, max=100)
        npu_output = self.npu_op_exec(npu_input1, bins=5000, min=0, max=100)
        cpu_output = cpu_output.astype(np.int32)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_histc_int32_5(self, device):
        npu_input1 = self.generate_single_data(0, 100, (1000,), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1.to(torch.float), bins=50)
        npu_output = self.npu_op_exec(npu_input1, bins=50)
        cpu_output = cpu_output.astype(np.int32)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_histc_float32_1(self, device):
        npu_input1 = self.generate_single_data(0, 100, (1000,), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, bins=50, min=0, max=100)
        npu_output = self.npu_op_exec(npu_input1, bins=50, min=0, max=100)
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test_histc_float32_2(self, device):
        npu_input1 = self.generate_single_data(0, 100, (20, 30, 2), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, bins=50, min=0, max=100)
        npu_output = self.npu_op_exec(npu_input1, bins=50, min=0, max=100)
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test_histc_float32_3(self, device):
        npu_input1 = self.generate_single_data(0, 100, (10000,), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, bins=50, min=0, max=100)
        npu_output = self.npu_op_exec(npu_input1, bins=50, min=0, max=100)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_histc_float32_4(self, device):
        npu_input1 = self.generate_single_data(0, 100, (1000,), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, bins=5000, min=0, max=100)
        npu_output = self.npu_op_exec(npu_input1, bins=5000, min=0, max=100)
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test_histc_float32_5(self, device):
        npu_input1 = self.generate_single_data(0, 100, (1000,), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, bins=50)
        npu_output = self.npu_op_exec(npu_input1, bins=50)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_histc_float16_1(self, device):
        npu_input1 = self.generate_single_data(0, 100, (1000,), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1.to(torch.float), bins=50, min=0, max=100)
        npu_output = self.npu_op_exec(npu_input1, bins=50, min=0, max=100)
        cpu_output = cpu_output.astype(np.float16)
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test_histc_float16_2(self, device):
        npu_input1 = self.generate_single_data(0, 100, (20, 30, 2), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1.to(torch.float), bins=50, min=0, max=100)
        npu_output = self.npu_op_exec(npu_input1, bins=50, min=0, max=100)
        cpu_output = cpu_output.astype(np.float16)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_histc_float16_3(self, device):
        npu_input1 = self.generate_single_data(0, 100, (10000,), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1.to(torch.float), bins=50, min=0, max=100)
        npu_output = self.npu_op_exec(npu_input1, bins=50, min=0, max=100)
        cpu_output = cpu_output.astype(np.float16)
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test_histc_float16_4(self, device):
        npu_input1 = self.generate_single_data(0, 100, (10000,), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1.to(torch.float), bins=5000, min=0, max=100)
        npu_output = self.npu_op_exec(npu_input1, bins=5000, min=0, max=100)
        cpu_output = cpu_output.astype(np.float16)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_histc_float16_5(self, device):
        npu_input1 = self.generate_single_data(0, 100, (1000,), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1.to(torch.float), bins=50)
        npu_output = self.npu_op_exec(npu_input1, bins=50)
        cpu_output = cpu_output.astype(np.float16)
        self.assertRtolEqual(cpu_output, npu_output)
        

instantiate_device_type_tests(TestHistc, globals(), except_for='cpu')  
if __name__ == '__main__':
    torch.npu.set_device("npu:1")
    run_tests()
