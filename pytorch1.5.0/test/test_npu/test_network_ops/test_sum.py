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

import sys
sys.path.append('..')
import torch
import torch.nn as nn
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestSum(TestCase):
    def cpu_op_exec(self, input1):
        output = input1.sum()
        output = output.numpy()
        return output
                  
    def npu_op_exec(self, input1):
        output = input1.sum()
        output = output.to("cpu")
        output = output.numpy()
        return output
        
    def cpu_op_exec_dim(self, input1, dim, dtype):
        output = torch.sum(input1, dim, keepdim=True, dtype=dtype)
        output = output.numpy()
        return output
                  
    def npu_op_exec_dim(self, input1, dim, dtype):
        output = torch.sum(input1, dim, keepdim=True, dtype=dtype)
        output = output.to("cpu")
        output = output.numpy()
        return output 

    def cpu_op_dim_exec_out(self, input1, dim, keepdim):
        out = torch.tensor(0).to(input1.dtype)
        torch.sum(input1, dim=dim, keepdim=keepdim, out=out)
        out = out.numpy()
        return out

    def npu_op_dim_exec_out(self, input1, dim, keepdim):
        out = torch.tensor(0).to(input1.dtype).npu()
        torch.sum(input1, dim=dim, keepdim=keepdim, out=out)
        out = out.to("cpu").numpy()
        output = torch.sum(input1, dim=dim, keepdim=keepdim)
        output = output.to("cpu").numpy()
        return out, output
   
    def sum_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, -1, 1)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)
            
    def sum_dim_result(self, shape_format):
        for item in shape_format:
            dim = np.random.randint(0, len(item[2]))
            cpu_input1, npu_input1 = create_common_tensor(item, -1, 1)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)

            cpu_out_dim = self.cpu_op_dim_exec_out(cpu_input1, dim=[0], keepdim=True)
            npu_out_dim, npu_output_dim = self.npu_op_dim_exec_out(npu_input1, dim=[0], keepdim=True)
            cpu_out_dim = cpu_out_dim.astype(npu_out_dim.dtype)
            if npu_out_dim.dtype != np.float16:
                self.assertRtolEqual(npu_out_dim, cpu_out_dim)
            else:
                self.assertRtolEqual(npu_out_dim, npu_output_dim)
                                
            cpu_output_dim = self.cpu_op_exec_dim(cpu_input1, dim, cpu_input1.dtype)
            npu_output_dim = self.npu_op_exec_dim(npu_input1, dim, npu_input1.dtype)
            cpu_output_dim = cpu_output_dim.astype(npu_output_dim.dtype)
            self.assertRtolEqual(cpu_output_dim, npu_output_dim)
            
    def test_sum_shape_format_fp16_1d(self, device):
        format_list = [0, 3]
        shape_format = [
            [np.float16, i, [18]] for i in format_list 
        ]
        self.sum_result(shape_format)
        
    def test_sum_shape_format_fp32_1d(self, device):
        format_list = [0, 3]
        shape_format = [
            [np.float32, i, [18]] for i in format_list 
        ]
        self.sum_result(shape_format)
        
    def test_sum_shape_format_fp16_2d(self, device):
        format_list = [0, 3, 29]
        shape_format = [
            [np.float16, i, [256, 1000]] for i in format_list 
        ]
        self.sum_result(shape_format)
        
    def test_sum_shape_format_fp32_2d(self, device):
        format_list = [0, 3,  29]
        shape_format = [
            [np.float32, i, [256, 1000]] for i in format_list 
        ]
        self.sum_result(shape_format)
        
    def test_sum_shape_format_fp16_3d(self, device):
        format_list = [0, 3, 29]
        shape_format = [
            [np.float16, i, [32, 48, 64]] for i in format_list 
        ]
        self.sum_result(shape_format)
        
    def test_sum_shape_format_fp32_3d(self, device):
        format_list = [0, 3,  29]
        shape_format = [
            [np.float32, i, [32, 48, 64]] for i in format_list 
        ]
        self.sum_result(shape_format)
        
    def test_sum_shape_format_fp16_4d(self, device):
        format_list = [0,  4, 29]
        shape_format = [
            [np.float16, i, [32, 24, 18, 18]] for i in format_list 
        ]
        self.sum_result(shape_format)
        
    def test_sum_shape_format_fp32_4d(self, device):
        format_list = [0, 3, 4, 29]
        shape_format = [
            [np.float32, i, [32, 24, 18, 18]] for i in format_list 
        ]
        self.sum_result(shape_format)
        
        # --------sum dim---------------------
        
    def test_sum_dim_shape_format_fp16_1d(self, device):
        format_list = [0, 3]
        shape_format = [
            [np.float16, i, [18]] for i in format_list 
        ]
        self.sum_dim_result(shape_format)
        
    def test_sum_dim_shape_format_fp32_1d(self, device):
        format_list = [0, 3]
        shape_format = [
            [np.float32, i, [18]] for i in format_list 
        ]
        self.sum_dim_result(shape_format)
        
    def test_sum_dim_shape_format_fp16_2d(self, device):
        format_list = [0, 3,  29]
        shape_format = [
            [np.float16, i, [256, 1000]] for i in format_list 
        ]
        self.sum_dim_result(shape_format)
        
    def test_sum_dim_shape_format_fp32_2d(self, device):
        format_list = [0, 3,  29]
        shape_format = [
            [np.float32, i, [256, 1000]] for i in format_list 
        ]
        self.sum_dim_result(shape_format)
        
    def test_sum_dim_shape_format_fp16_3d(self, device):
        # TODO(ascend): Insufficient precision
        #format=29精度不满足 format_list = [0, 3,  29]
        format_list = [0, 3]
        shape_format = [
            [np.float16, i, [32, 48, 64]] for i in format_list 
        ]
        self.sum_dim_result(shape_format)
        
    def test_sum_dim_shape_format_fp32_3d(self, device):
        format_list = [0, 3,  29]
        shape_format = [
            [np.float32, i, [32, 48, 64]] for i in format_list 
        ]
        self.sum_dim_result(shape_format)
        
    def test_sum_dim_shape_format_fp16_4d(self, device):
        format_list = [0, 3]
        shape_format = [
            [np.float16, i, [16, 16, 9, 9]] for i in format_list 
        ]
        self.sum_dim_result(shape_format)
        
    def test_sum_dim_shape_format_fp32_4d(self, device):
        format_list = [0, 3, 4]
        shape_format = [
            [np.float32, i, [32, 24, 18, 18]] for i in format_list 
        ]
        self.sum_dim_result(shape_format)

    def test_sum_dim_with_zero_shape_format(self, device):
        format_list = [0, 3, 4]
        shape_format = [
            [np.float32, i, [2, 0, 3]] for i in format_list 
        ]
        self.sum_dim_result(shape_format)
        self.sum_result(shape_format)


instantiate_device_type_tests(TestSum, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
