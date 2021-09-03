# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
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

class TestInstanceNorm(TestCase):

    def generate_data(self, min, max, shape, dtype): 
        x = np.random.uniform(min, max, shape).astype(dtype) 
        w = np.random.uniform(min, max, shape).astype(dtype) 
        b = np.random.uniform(min, max, shape).astype(dtype) 
        rm = np.random.uniform(min, max, shape).astype(dtype) 
        rv = np.random.uniform(min, max, shape).astype(dtype) 
     
        #modify from numpy.ndarray to torch.tensor 
        npu_x = torch.from_numpy(x) 
        npu_w = torch.from_numpy(w) 
        npu_b = torch.from_numpy(b) 
        npu_rm = torch.from_numpy(rm)
        npu_rv = torch.from_numpy(rv) 
         
        return npu_x, npu_w,npu_b,npu_rm,npu_rv
         
    def generate_single_data(self, min, max, shape, dtype): 
        input1 = np.random.uniform(min, max, shape).astype(dtype) 
        npu_input1 = torch.from_numpy(input1) 
         
        return npu_input1 
     
     
    def generate_three_data(self, min, max, shape, dtype): 
        input1 = np.random.uniform(min, max, shape).astype(dtype) 
        input2 = np.random.uniform(min, max, shape).astype(dtype) 
        input3 = np.random.uniform(min, max, shape).astype(dtype) 
     
        #modify from numpy.ndarray to torch.tensor 
        npu_input1 = torch.from_numpy(input1) 
        npu_input2 = torch.from_numpy(input2) 
        npu_input3 = torch.from_numpy(input3) 
         
        return npu_input1, npu_input2, npu_input3 
     
     
    def cpu_op_exec(self, x, w,b,rm,rv,use_input_stats, momentum, eps): 
        axis = []
        for i in range(2,len(x.shape)):
            axis.append(i)
        mean = np.mean(x, tuple(axis), keepdims=True)
        var = np.var(x, tuple(axis), keepdims=True)

        if input_use ==True:
            mean = (mean-momentum*mean) + momentum*rm
            var = (var-momentum*var) + momentum*rv
            print("11")
            y = (x - mean)/np.sqrt(var + eps)
            output = w*y + b
        else:
            y = (x - mean)/np.sqrt(var + eps)
            output = w*y + b
        output = output.numpy() 
        return output 
     
     
    def npu_op_exec(self, x, w,b,rm,rv,use_input_stats, momentum, eps): 
        x = x.to("npu") 
        w = w.to("npu") 
        b = b.to("npu") 
        rm = rm.to("npu") 
        rv = rv.to("npu") 
        axis = []
        for i in range(2,len(x.shape)):
            axis.append(i)
        mean = np.mean(x, tuple(axis), keepdims=True)
        var = np.var(x, tuple(axis), keepdims=True)

        if input_use ==True:
            mean = (mean-momentum*mean) + momentum*rm
            var = (var-momentum*var) + momentum*rv
            print("11")
            y = (x - mean)/np.sqrt(var + eps)
            output = w*y + b
        else:
            y = (x - mean)/np.sqrt(var + eps)
            output = w*y + b
        output = output.to("cpu") 
        output = output.numpy() 
        return output 
         
     
    def npu_op_exec_scalar(self, input1, input2): 
        input1 = input1.to("npu") 
        output = input1 + input2 
        output = output.to("cpu") 
        output = output.numpy() 
        return output 
     
     
    def npu_op_exec_out(self, input1, input2, input3): 
        input1 = input1.to("npu") 
        input2 = input2.to("npu") 
        output = input3.to("npu") 
        torch.add(input1, input2, out=output) 
        output = output.to("cpu") 
        output = output.numpy() 
        return output 
         
    def test_add_float16(self, device):
        npu_x, npu_w,npu_b,npu_rm,npu_rv = self.generate_data(0, 100, (5, 6, 7), np.float16) 
        cpu_output = self.cpu_op_exec(npu_x, npu_w,npu_b,npu_rm,npu_rv,True, 0.1, 0.00001) 
        npu_output = self.npu_op_exec(npu_x, npu_w,npu_b,npu_rm,npu_rv,True, 0.1, 0.00001) 
        self.assertRtolEqual(cpu_output, npu_output)
     
     
    def test_add_float32(self, device):
        npu_x, npu_w,npu_b,npu_rm,npu_rv = self.generate_data(0, 100, (5, 6, 7), np.float32) 
        cpu_output = self.cpu_op_exec(npu_x, npu_w,npu_b,npu_rm,npu_rv,True, 0.1, 0.00001) 
        npu_output = self.npu_op_exec(npu_x, npu_w,npu_b,npu_rm,npu_rv,True, 0.1, 0.00001) 
        self.assertRtolEqual(cpu_output, npu_output)
     
     
    def test_add_float32_out(self, device):
        npu_input1, npu_input2, npu_input3  = generate_three_data(0, 100, (4,3), np.float32) 
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2) 
        npu_output = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3) 
        self.assertRtolEqual(cpu_output, npu_output)
     
     
    def test_add_float32_broadcast(self, device):
        npu_input1 = self.generate_single_data(0, 100, (4,3,1), np.float32) 
        npu_input2 = self.generate_single_data(0, 100, (4,1,5), np.float32) 
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2) 
        npu_output = self.npu_op_exec(npu_input1, npu_input2) 
        self.assertRtolEqual(cpu_output, npu_output)
     
     
    def test_add_int32(self, device):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2,3), np.int32) 
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2) 
        npu_output = self.npu_op_exec(npu_input1, npu_input2) 
        self.assertRtolEqual(cpu_output, npu_output)
     
     
    def test_add_scalar_float32(self, device):
        npu_input1, _= self.generate_data(0, 100, (2,3), np.float32) 
        cpu_output = self.cpu_op_exec(npu_input1, 1) 
        npu_output = self.npu_op_exec_scalar(npu_input1, 1) 
        self.assertRtolEqual(cpu_output, npu_output)
     
     
    def npu_uncontiguous_op_exec_scalar(self, input1, input2): 
        input1 = input1.to("npu") 
        input1 = input1.as_strided([2,2], [1,2], 1) 
        output = torch.add(input1, input2) 
        output = output.to("cpu") 
        output = output.numpy() 
        return output 
         
    def cpu_uncontiguous_op_exec_scalar(self, input1, input2): 
        input1 = input1.as_strided([2,2], [1,2], 1) 
        output = torch.add(input1, input2) 
        output = output.numpy() 
        return output 
         
    def test_add_uncontiguous_float32_scalar(self, device):
        npu_input1, npu_input2 = self.generate_data(0, 100, (4,3), np.float32) 
        cpu_input1 = copy.deepcopy(npu_input1) 
        cpu_output = self.cpu_uncontiguous_op_exec_scalar(cpu_input1, 2) 
        npu_output = self.npu_uncontiguous_op_exec_scalar(npu_input1, 2) 
        self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestInstanceNorm, globals(), except_for='cpu')     
if __name__ == '__main__': 
    torch.npu.set_device("npu:2") 
    run_tests()

