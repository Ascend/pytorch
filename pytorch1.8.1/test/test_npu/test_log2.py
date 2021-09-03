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

class TestLog2(TestCase):
    
    def cpu_op_exec(self,input1):
        output = torch.log2(input1)
        output = output.numpy()
        return output
        
    def npu_op_exec(self,input1):
        output = torch.log2(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output
        
    def cpu_op_exec_(self,input1):
        output = torch.log2_(input1)
        output = input1.numpy()
        return output
            
    def npu_op_exec_(self,input1):
        output = torch.log2_(input1)
        output = input1.to("cpu")
        output = output.numpy()
        return output
    
    def cpu_op_exec_out(self,input1,cpu_out):
        output = torch.log2(input1, out = cpu_out)
        output = cpu_out.numpy()
        return output
        
    def npu_op_exec_out(self,input1,npu_out):
        output = torch.log2(input1, out = npu_out)
        output = npu_out.to("cpu")
        output = output.numpy()
        return output
        
    def test_log2_float32_common_shape_format(self, device):
        shape_format = [
                [[np.float32, -1, (1)]], 
                [[np.float32, -1, (4, 23)]],
                [[np.float32, -1, (2, 3)]],
                [[np.float32, -1, (12, 23)]]
        ]
        for item in shape_format:            
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)
            
    def test_log2_float321_common_shape_format(self, device):
        shape_format = [
                [[np.float32, -1, (3)]], 
                [[np.float32, -1, (4, 3)]],
                [[np.float32, -1, (12, 32)]],
                [[np.float32, -1, (22, 38)]]
        ]
        for item in shape_format:       
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec_(cpu_input1)
            npu_output = self.npu_op_exec_(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)
	
    def test_log2_out_float32_common_shape_format(self, device):
        shape_format = [
                [[np.float32, -1, (4)]], 
                [[np.float32, -1, (4, 1, 5)]],
                [[np.float32, -1, (2, 3, 8)]],
                [[np.float32, -1, (2, 13, 56)]]
        ]
        for item in shape_format:          
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_out, npu_out = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec_out(cpu_input1,cpu_out)
            npu_output = self.npu_op_exec_out(npu_input1,npu_out)
            self.assertRtolEqual(cpu_output, npu_output)
  
instantiate_device_type_tests(TestLog2, globals(), except_for="cpu")

if __name__ == "__main__":
    torch.npu.set_device("npu:6")
    run_tests()