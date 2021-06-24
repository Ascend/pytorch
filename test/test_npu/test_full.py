# Copyright (c) 2020 Huawei Technologies Co., Ltd
 
# All rights reserved.
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


class TestFull(TestCase):

    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype)

        #modify from numpy.ndarray to torch.tensor
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
        
        return npu_input1, npu_input2
        

    def generate_single_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        
        return npu_input1

    def generate_three_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input3 = np.random.uniform(min_d, max_d, shape).astype(dtype)

        #modify from numpy.ndarray to torch.tensor
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
        npu_input3 = torch.from_numpy(input3)
        
        return npu_input1, npu_input2, npu_input3

    def generate_scalar(self, min_d, max_d):
        scalar = np.random.uniform(min_d, max_d)
        return scalar

    def generate_int_scalar(self, min_d, max_d):
        scalar = np.random.randint( min_d, max_d)
        return scalar

    def cpu_op_exec(self, sizes, value):
        output = torch.full(sizes, value)
        output = output.numpy()
        return output

    def npu_op_exec(self, sizes, value):
        #input1 = input1.to("npu")
        output = torch.full(sizes, value, device="npu")
        output = output.to("cpu")
        output = output.numpy()
        return output    
    
    def cpu_op_name_exec(self,sizes, value, names):
        out = torch.full(sizes, value,names=names)
        output = out.numpy()
        return output


    def npu_op_name_exec(self, sizes, value, names):
        out1 = torch.full(sizes, value, names=names, device="npu")
        output = out1.to("cpu")
        output = output.numpy()
        return output


    def cpu_op_out_exec(self, sizes, value, out):
        torch.full(sizes, value, out=out)
        output = out.numpy()
        return output


    def npu_op_out_exec(self, sizes, value, out):
       # input1 = input1.to("npu")
        out = out.to("npu")
        torch.full(sizes, value, out=out)
        output = out.to("cpu")
        output = output.numpy()
        return output

    def test_full_common_shape_format(self, device):
        shape_format =[ 
                [[np.float32, -1, (4, 3)], (2,3), 4],
                [[np.float32, 0, (4, 3, 1)], (5,6,7), 5],
                [[np.float16, 3, (2, 3)], (2,3), 6],
                [[np.float16, 4, (2, 3)], (2,3), 6],
                [[np.int32, 0  , (4, 3)], (2,3, 3,5), 7],
                [[np.int32, 3  , (1,2,1,4, 3)], (1,1,1,3,5), 7],
              #  [[np.int32, 4 , (4, 3)], (3,5), 7]
        ]
        for item in shape_format:
           # cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec(item[1], item[2])
            npu_output = self.npu_op_exec(item[1], item[2])
            self.assertRtolEqual(cpu_output, npu_output) 
    
    def test_full_name_common_shape_format(self, device):
        shape_format = [
                [[np.float32, -1, (4, 3)], (1,2,3), 4,( "channal","rows", "cols")],
                [[np.float32, 4, (4, 3, 1)], (1,5,6,7), 5,("batch", "channal","rows", "cols")],
                [[np.float16, 3, (2, 3)], (2,3), 6, ("rows", "cols")],
                [[np.int32,   0, (4, 3, 1)], (2,3,5), 7, ("channal","rows", "cols")]
        ]
        
        for item in shape_format:
           # cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_name_exec(item[1], item[2],item[3])
            npu_output = self.npu_op_name_exec(item[1], item[2],item[3])
            self.assertRtolEqual(cpu_output, npu_output)
    
    def test_full_out_common_shape_format(self, device):
        shape_format = [
                [[np.float32, -1, (4, 3)], (4,3), 4],
                [[np.float32, -1, (5,6,7)], (5,6,7), 5],            
                [[np.int32,   -1, (2,3, 5)], (2,3,5), 7]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_out_exec(item[1], item[2], cpu_input1)
            npu_output = self.npu_op_out_exec(item[1], item[2], npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)
    

instantiate_device_type_tests(TestFull, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:6")
    run_tests()
