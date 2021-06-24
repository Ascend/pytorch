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
import random
import math

class TestLerp(TestCase):
# pylint: disable=unused-variable,unused-argument

    def cpu_op_exec(self, input1, input2, input3): 
        output = torch.lerp(input1,input2,input3) 
        output = output.numpy() 
        return output 
    
    def npu_op_exec(self, input1, input2, input3): 
        output = torch.lerp(input1, input2, input3) 
        output = output.to("cpu") 
        output = output.numpy() 
        return output

    def cpu_op_out_exec(self, input1, input2, input3):
        output = torch.ones_like(input1) 
        torch.lerp(input1,input2,input3, out = output) 
        output = output.numpy() 
        return output 
    
    def npu_op_out_exec(self, input1, input2, input3):
        output = torch.ones_like(input1) 
        torch.lerp(input1, input2, input3, out = output) 
        output = output.to("cpu") 
        output = output.numpy() 
        return output

    def cpu_op_scalar_out_exec(self, input1, input2, input3):
        output = torch.ones_like(input1) 
        torch.lerp(input1,input2,input3, out = output) 
        output = output.numpy() 
        return output 
    
    def npu_op_scalar_out_exec(self, input1, input2, input3):
        output = torch.ones_like(input1) 
        torch.lerp(input1, input2, input3, out = output) 
        output = output.to("cpu") 
        output = output.numpy() 
        return output


    def test_lerp_common_shape_format(self, device):
        shape_format = [
            [[np.float32, -1, (4, 2, 2, 3)]],
            [[np.float32, -1, (2, 2, 3, 4)]],
            [[np.float32, -1, (3, 3, 3)]],
            [[np.float32, -1, (4, 4, 4)]]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 1, 100)
            cpu_input3, npu_input3 = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, cpu_input3)
            npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_input3)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_lerp_float16_shape_format(self, device):
        def cpu_op_exec_fp16(input1, input2, input3):
            input1 = input1.to(torch.float32)
            input2 = input2.to(torch.float32)
            input3 = input3.to(torch.float32)
            output = torch.lerp(input1,input2,input3)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        shape_format = [
            [[np.float16, -1, (100, 4, 5, 5)]],
            [[np.float16, -1, (100, 5, 5, 4)]],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 10, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 10, 100)
            cpu_input3, npu_input3 = create_common_tensor(item[0], 10, 100)
            cpu_output = cpu_op_exec_fp16(cpu_input1, cpu_input2, cpu_input3)
            npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_input3)
            self.assertRtolEqual(cpu_output, npu_output, prec=0.003, prec16=0.003)

    
    def test_lerp_out_common_shape_format(self, device):
        shape_format = [
            [[np.float32, -1, (4, 2, 2, 3)]],
            [[np.float32, -1, (2, 2, 3, 4)]],
            [[np.float32, -1, (3, 3, 3)]],
            [[np.float32, -1, (4, 4, 4)]]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 1, 100)
            cpu_input3, npu_input3 = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_out_exec(cpu_input1, cpu_input2, cpu_input3)
            npu_output = self.npu_op_out_exec(npu_input1, npu_input2, npu_input3)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_lerp_out_float16_shape_format(self, device):
        def cpu_op_out_exec_fp16(input1, input2, input3):
            input1 = input1.to(torch.float32)
            input2 = input2.to(torch.float32)
            input3 = input3.to(torch.float32)
            output = torch.ones_like(input1)
            torch.lerp(input1,input2,input3, out = output)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        shape_format = [
            [[np.float16, -1, (100, 4, 5, 5)]],
            [[np.float16, -1, (100, 5, 5, 4)]],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 10, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 10, 100)
            cpu_input3, npu_input3 = create_common_tensor(item[0], 10, 100)
            cpu_output = cpu_op_out_exec_fp16(cpu_input1, cpu_input2, cpu_input3)
            npu_output = self.npu_op_out_exec(npu_input1, npu_input2, npu_input3)
            self.assertRtolEqual(cpu_output, npu_output, prec=0.003, prec16=0.003)

    def test_lerp_scalar_common_shape_format(self, device):
        shape_format = [
            [[np.float32, -1, (4, 2, 2, 3)], 1.0],
            [[np.float32, -1, (2, 2, 3, 4)], 2.0],
            [[np.float32, -1, (3, 3, 3)], 1.2],
            [[np.float32, -1, (4, 4, 4)], 1.2]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 1, 100)
            cpu_input3 = item[1]
            npu_input3 = item[1]
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, cpu_input3)
            npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_input3)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_lerp_scalar_float16_shape_format(self, device):
        def cpu_op_scalar_exec_fp16(input1, input2, input3):
            input1 = input1.to(torch.float32)
            input2 = input2.to(torch.float32)
            output = torch.lerp(input1,input2,input3)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        shape_format = [
            [[np.float16, -1, (100, 4, 5, 5)], 1,2],
            [[np.float16, -1, (100, 5, 5, 4)], 1.2],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 10, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 10, 100)
            cpu_input3 = item[1]
            npu_input3 = item[1]
            cpu_output = cpu_op_scalar_exec_fp16(cpu_input1, cpu_input2, cpu_input3)
            npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_input3)
            self.assertRtolEqual(cpu_output, npu_output, prec=0.003, prec16=0.003)

    
    def test_lerp_scalar_out_common_shape_format(self, device):
        shape_format = [
            [[np.float32, -1, (4, 2, 2, 3)], 1.2],
            [[np.float32, -1, (2, 2, 3, 4)],1.2],
            [[np.float32, -1, (3, 3, 3)], 1.0],
            [[np.float32, -1, (4, 4, 4)], 2.0]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 1, 100)
            cpu_input3, npu_input3 = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, cpu_input3)
            npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_input3)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_lerp_scalar_out_float16_shape_format(self, device):
        def cpu_op_scalar_out_exec_fp16(input1, input2, input3):
            input1 = input1.to(torch.float32)
            input2 = input2.to(torch.float32)
            output = torch.ones_like(input1)
            torch.lerp(input1,input2,input3, out = output)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        shape_format = [
            [[np.float16, -1, (100, 4, 5, 5)], 1.2],
            [[np.float16, -1, (100, 5, 5, 4)], 1.2],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 10, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 10, 100)
            cpu_input3 = item[1]
            npu_input3 = item[1]
            cpu_output = cpu_op_scalar_out_exec_fp16(cpu_input1, cpu_input2, cpu_input3)
            npu_output = self.npu_op_scalar_out_exec(npu_input1, npu_input2, npu_input3)
            self.assertRtolEqual(cpu_output, npu_output, prec=0.003, prec16=0.003)
     
instantiate_device_type_tests(TestLerp, globals(), except_for='cpu')
if __name__ == '__main__': 
    run_tests() 
