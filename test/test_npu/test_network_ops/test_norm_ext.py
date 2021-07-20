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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestNorm(TestCase):
    def norm_output_size(self, input, dim, keepdim):
        output_size = list(input.size())
        for i in dim:
          if i < 0:
            i = i + input.dim()
          if i < input.dim() and keepdim == True:
            output_size[i] = 1
          if  i < input.dim() and keepdim == False:
            output_size.pop(i)  
        return output_size

    def cpu_out_exec(self, input, p1, dim1, keepdim1, dtype1):
        output_size = self.norm_output_size(input, dim1, keepdim1)
        cpu_out = torch.randn(output_size)
        output = torch.norm(input, p = p1, dim = dim1 , keepdim = keepdim1, out = cpu_out, dtype = dtype1)
        return output
    
    def npu_out_exec(self, input, p1, dim1, keepdim1, dtype1):
        output_size = self.norm_output_size(input, dim1, keepdim1)
        npu_out = torch.randn(output_size).npu().to(input.dtype)
        output1 = torch.norm(input, p = p1, dim = dim1 , keepdim = keepdim1, out = npu_out, dtype = input.dtype)
        output = output1.to("cpu")
        return output
        
    def test_norm_shape_format_0(self, device):
        shape_format = [
                [[np.float16, 0, (1)]],
                [[np.float32, 0, (1)]],
        ] 
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_out_exec(cpu_input, 0, [0], True, torch.float)
            npu_output = self.npu_out_exec(npu_input, 0, [0], True, torch.float)
            cpu_output = cpu_output.to(npu_output.dtype)
            self.assertRtolEqual(cpu_output.numpy(), npu_output.numpy())
       
    def test_norm_shape_format_1(self, device):
        shape_format = [
                [[np.float16, 0, (12, 33)]],
                [[np.float32, 0, (12, 33)]],
        ] 
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_out_exec(cpu_input, 1, [0,1], True, torch.float)
            npu_output = self.npu_out_exec(npu_input, 1, [0,1], True, torch.float)
            cpu_output = cpu_output.to(npu_output.dtype)
            self.assertRtolEqual(cpu_output.numpy(), npu_output.numpy())
            
    def test_norm_shape_format_2(self, device):
        shape_format = [
                # [[np.float16, 0, (12, 33)]],  # result error
                [[np.float32, 0, (12, 33)]],
        ] 
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_out_exec(cpu_input, 2, [0], False, torch.float)
            npu_output = self.npu_out_exec(npu_input, 2, [0], False, torch.float)
            npu_output = npu_output.to(cpu_output.dtype)
            self.assertRtolEqual(cpu_output.numpy(), npu_output.numpy())
            
    def test_norm_shape_format_3(self, device):
        shape_format = [
                [[np.float16, 0, (10, 24, 56, 2048)]], # result error
                [[np.float32, 0, (10, 24, 56, 2048)]],
        ] 
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_out_exec(cpu_input, 3, [1,2], True, torch.float)
            npu_output = self.npu_out_exec(npu_input, 3, [1,2], True, torch.float)
            cpu_output = cpu_output.to(npu_output.dtype)
            self.assertRtolEqual(cpu_output.numpy(), npu_output.numpy())
           
    def test_norm_shape_format_inf(self, device):
        shape_format = [
                [[np.float16, 0, (64, 64, 64, 64)]],
                [[np.float32, 0, (64, 64, 64, 64)]],
        ] 
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_out_exec(cpu_input, float("inf"), [1,2], True, torch.float)
            npu_output = self.npu_out_exec(npu_input, float("inf"), [1,2], True, torch.float)
            cpu_output = cpu_output.to(npu_output.dtype)
            self.assertRtolEqual(cpu_output.numpy(), npu_output.numpy())
           
    def test_norm_shape_format_inf1(self, device):
        shape_format = [
                [[np.float16, 0, (64, 64, 64, 64)]],
                [[np.float32, 0, (64, 64, 64, 64)]],
        ] 
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            cpu_output = self.cpu_out_exec(cpu_input, float("-inf"), [1,2], False, torch.float)
            npu_output = self.npu_out_exec(npu_input, float("-inf"), [1,2], False, torch.float)
            cpu_output = cpu_output.to(npu_output.dtype)
            self.assertRtolEqual(cpu_output.numpy(), npu_output.numpy())
     
instantiate_device_type_tests(TestNorm, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()