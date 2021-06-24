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

class Testixor(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        
        #modify from numpy.ndarray to torch.tensor
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
    
        return npu_input1, npu_input2
        
    def generate_bool_data(self, min_d, max_d, shape):
        input1 = np.random.uniform(min_d, max_d, shape)
        input2 = np.random.uniform(min_d, max_d, shape)
        input1 = input1.reshape(-1)
        input2 = input2.reshape(-1)
        for i in range(len(input1)):
            if input1.any() < 0.5:
                input1[i] = 0
        for i in range(len(input2)):
            if input2.any() < 0.5:
                input2[i] = 0
        input1 = input1.astype(np.bool)
        input2 = input2.astype(np.bool)
        input1 = input1.reshape(shape)
        input2 = input2.reshape(shape)
        #modify from numpy.ndarray to torch.tensor
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
    
        return npu_input1, npu_input2
        
    def generate_single_data(self, min_d, max_d, shape, dtype): 
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype) 
        npu_input1 = torch.from_numpy(input1) 
         
        return npu_input1 
     
    def generate_single_bool_data(self, min_d, max_d, shape):
        input1 = np.random.uniform(min_d, max_d, shape)
        input1 = input1.reshape(-1)
        for i in range(len(input1)):
            if input1[i] < 0.5:
                input1[i] = 0
        input1 = input1.astype(np.bool)
        input1 = input1.reshape(shape)
        npu_input1 = torch.from_numpy(input1)
        return npu_input1
     
    def generate_three_data(self, min_d, max_d, shape, dtype): 
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype) 
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype) 
        input3 = np.random.uniform(min_d, max_d, shape).astype(dtype) 

        npu_input1 = torch.from_numpy(input1) 
        npu_input2 = torch.from_numpy(input2) 
        npu_input3 = torch.from_numpy(input3) 
         
        return npu_input1, npu_input2, npu_input3 
    
    def npu_op_exec_out(self, input1, input2, input3): 
        input1 = input1.to("npu") 
        input2 = input2.to("npu") 
        output = input3.to("npu") 
        input1.__ixor__(input2, out=output) 
        output = output.to("cpu") 
        output = output.numpy() 
        return output 
     
    def cpu_op_exec_out(self, input1, input2, input3): 
        output = input3
        input1.__ixor__(input2, out=output) 
        output = output.numpy() 
        return output 
        
    def npu_op_exec_scalar_out(self, input1, input2, input3): 
        output = input3.to("npu")
        input1 = input1.to("npu")
        input2 = torch.tensor(input2)
        input2 = input2.to("npu") 
        input1.__ixor__(input2, out=output) 
        output = output.to("cpu") 
        output = output.numpy() 
        return output 
     
    def test__ixor__int32(self, device): 
        npu_input1, npu_input2 = self.generate_data(0, 100, (2,3), np.int32) 
        cpu_output = self.cpu_op_exec_out(npu_input1, npu_input2,npu_input1) 
        npu_output = self.npu_op_exec_out(npu_input1, npu_input2,npu_input1) 
        self.assertRtolEqual(cpu_output, npu_output) 

    def test__ixor__int32_scalar(self, device):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2,3), np.int32)
        cpu_output = self.cpu_op_exec_out(npu_input1, 1, npu_input1)
        npu_output = self.npu_op_exec_scalar_out(npu_input1, 1, npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test__ixor__float32_out(self, device):
        npu_input1, npu_input2, npu_input3 = self.generate_three_data(0, 100, (4, 3), np.int32)
        cpu_output = self.cpu_op_exec_out(npu_input1, npu_input2, npu_input3)
        npu_output = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3)
        self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(Testixor, globals(), except_for='cpu')     
if __name__ == '__main__': 
    run_tests() 
