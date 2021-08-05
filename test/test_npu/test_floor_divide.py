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
import random
import math
 
class TestFloorDivide(TestCase):    
# pylint: disable=unused-variable,unused-argument

    def cpu_op_exec(self, input1, input2): 
        output = torch.floor_divide(input1,input2) 
        output = output.numpy() 
        return output 
    
    def cpu_op_exec_fp16(self, input1, input2):
        input1 = input1.to(torch.float32)
        input2 = input2.to(torch.float32)
        output = torch.floor_divide(input1, input2)
        output = output.numpy()
        output = output.astype(np.float16)
        return output
    
    def npu_op_exec(self, input1, input2): 
        input1 = input1.to("npu") 
        input2 = input2.to("npu") 
        output = torch.floor_divide(input1,input2) 
        output = output.to("cpu") 
        output = output.numpy() 
        return output 
    
    def test_floor_divide_common_shape_format(self, device):
        shape_format = [
            [[np.float32, -1, (4, 3, 3)]],
            [[np.float32, -1, (4, 5, 5)]],
            [[np.float32, -1, (3, 3, 3)]],
            [[np.float32, -1, (4, 4, 4)]],
            [[np.float32, -1, (2, 0, 2)]]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 10, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 10, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_floor_divide_float16_shape_format(self, device):
        shape_format = [
            [[np.float16, -1, (4, 2, 6, 6)]],
            [[np.float16, -1, (4, 2, 8, 8)]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec_fp16(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_floor_divide_int32_shape_format(self, device):
        shape_format = [
            [[np.int32, -1, (4, 3)]],
            [[np.int32, -1, (4, 5)]],
            [[np.int32, -1, (3, 3)]],
            [[np.int32, -1, (4, 4)]]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 100, 1000)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 100, 1000)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_floor_divide_int8_shape_format(self, device):
        shape_format = [
            [[np.int8, -1, (4, 8, 3)]],
            [[np.int8, -1, (4, 7, 5)]],
            [[np.int8, -1, (3, 6, 3)]],
            [[np.int8, -1, (4, 5, 4)]]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 10, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 10, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_floor_divide_uint8_shape_format(self, device):
        shape_format = [
            [[np.uint8, -1, (4, 3, 3)]],
            [[np.uint8, -1, (4, 5, 5)]],
            [[np.uint8, -1, (3, 3, 3)]],
            [[np.uint8, -1, (4, 4, 4)]]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 10, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 10, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestFloorDivide, globals(), except_for='cpu')
if __name__ == '__main__':  
    run_tests() 