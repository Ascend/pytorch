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

# coding: utf-8

import torch
import numpy as np
import sys
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestExpm1(TestCase):

    def cpu_op_exec(self,input1):
        output = torch.expm1(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self,input1):
        output = torch.expm1(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_(self,input1):
        torch.expm1_(input1)
        output = input1.numpy()
        return output

    def npu_op_exec_(self,input1):
        torch.expm1_(input1)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_out(self,input1,out):
        torch.expm1(input1, out=out)
        output = out.numpy()
        return output

    def npu_op_exec_out(self,input1, out):
        torch.expm1(input1, out=out)
        output = out.to("cpu")
        output = output.numpy()
        return output
    def test_expm1_float32_common_shape_format(self, device):
        shape_format = [
                [np.float32, -1 , (4, 3)],
                [np.float32, -1, (2,4, 3)],
                [np.float32, 3, (20, 13)],
                [np.float32, 4, (20, 13)],
                [np.float32, 29, (20, 13)]
                ]
        for item in shape_format:            
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 10)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)
    		
    def test_expm1_float321_common_shape_format(self, device):
        shape_format = [
                [np.float32, -1, (4, 3)],
                [np.float32, 0 , (4, 3)],
                [np.float32, -1, (2,4, 3)],
                [np.float32, 3, (20, 13)],
                [np.float32, 4, (20, 13)],
                [np.float32, 29, (20, 13)]
        ]
        for item in shape_format:        
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 10)
            cpu_output = self.cpu_op_exec_(cpu_input1)
            npu_output = self.npu_op_exec_(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)
	
    def test_expm1_out_float32_common_shape_format(self, device):
        shape_format = [
                [np.float32, -1, (4, 3)],
                [np.float32, 0 , (4, 3)],
                [np.float32, -1, (2,4, 3)],
                [np.float32, 3, (20, 13)],
                [np.float32, 4, (20, 13)],
                [np.float32, 29, (20, 13)]        
        ]
        for item in shape_format:          
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 10)
            cpu_out, npu_out = create_common_tensor(item, 1, 10)
            cpu_output = self.cpu_op_exec_out(cpu_input1,cpu_out)
            npu_output = self.npu_op_exec_out(npu_input1,npu_out)
            self.assertRtolEqual(cpu_output, npu_output)
    
    def test_expm1_float16_common_shape_format(self, device):
        shape_format = [
                [np.float16, -1 , (4, 3)],
                [np.float16, -1, (2,4, 3)],
                [np.float16, -1, (100, 20, 10)],
                [np.float16, 3, (20, 13)],
                [np.float16, 4, (20, 13)],
                [np.float16, 29, (20, 13)]
                ]
        for item in shape_format:            
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 10)
            if item[0] == np.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            if item[0] == np.float16:
                cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)
    		
    def test_expm1_float16__common_shape_format(self, device):
        shape_format = [
                [np.float16, -1, (4, 3)],
                [np.float16, 0 , (4, 3)],
                [np.float16, -1, (2,4, 3)],
                [np.float16, -1, (100, 20, 10)],
                [np.float16, 3, (20, 13)],
                [np.float16, 4, (20, 13)],
                [np.float16, 29, (20, 13)]
        ]
        for item in shape_format:        
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 10)
            if item[0] == np.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec_(cpu_input1)
            npu_output = self.npu_op_exec_(npu_input1)
            if item[0] == np.float16:
                cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)
	
    def test_expm1_out_float16_common_shape_format(self, device):
        shape_format = [
                [np.float16, -1, (4, 3)],
                [np.float16, 0 , (4, 3)],
                [np.float16, -1, (2,4, 3)],
                [np.float16, -1, (100, 20, 10)],
                [np.float16, 3, (20, 13)],
                [np.float16, 4, (20, 13)],
                [np.float16, 29, (20, 13)]        
        ]
        for item in shape_format:          
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 10)
            cpu_out, npu_out = create_common_tensor(item, 1, 10)
            if item[0] == np.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_out = cpu_out.to(torch.float32)
            cpu_output = self.cpu_op_exec_out(cpu_input1,cpu_out)
            npu_output = self.npu_op_exec_out(npu_input1,npu_out)
            if item[0] == np.float16:
                cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestExpm1, globals(), except_for="cpu")

if __name__ == "__main__":
    run_tests()