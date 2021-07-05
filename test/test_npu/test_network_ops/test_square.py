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

class TestSquare(TestCase):     
# pylint: disable=unused-variable,unused-argument
    def cpu_op_exec(self, input1): 
        flag = 0
        if input1.dtype == torch.float16:
            input1 = input1.to(torch.float32)
            flag = 1
        output = torch.square(input1)
        if flag == 1:
            output = output.to(torch.float16)
        output = output.numpy() 
        return output 

    def npu_op_exec(self, input1): 
        input1 = input1.to("npu") 
        output = torch.square(input1)
        output = output.to("cpu")  
        output = output.numpy() 
        return output

    def cpu_op_inplace_exec(self, input1): 
        flag = 0
        if input1.dtype == torch.float16:
            input1 = input1.to(torch.float32)
            flag = 1
        input1.square_()
        if flag == 1:
            input1 = input1.to(torch.float16)
        output = input1.numpy() 
        return output 

    def npu_op_inplace_exec(self, input1): 
        input1 = input1.to("npu") 
        input1.square_()
        output = input1.to("cpu") 
        output = output.numpy() 
        return output 

    def test_square_common_shape_format(self, device):
        shape_format = [
            [[np.float32, -1, (4, 3, 3)]],
            [[np.float32, -1, (4, 5, 5)]],
            [[np.float32, -1, (3, 3, 3)]],
            [[np.float32, -1, (4, 4, 4)]]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 10)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

            cpu_output = self.cpu_op_inplace_exec(cpu_input1)
            npu_output = self.npu_op_inplace_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)
    
    def test_square_int32_shape_format(self, device):
        shape_format = [
            [[np.int32, -1, (4, 2)]],
            [[np.int32, -1, (4, 2)]],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -10, 10)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

            cpu_output = self.cpu_op_inplace_exec(cpu_input1)
            npu_output = self.npu_op_inplace_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_square_float16_shape_format(self, device):
        shape_format = [
            [[np.float16, -1, (4, 2, 6, 6)]],
            [[np.float16, -1, (4, 2, 8, 8)]],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

            cpu_output = self.cpu_op_inplace_exec(cpu_input1)
            npu_output = self.npu_op_inplace_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)
    
instantiate_device_type_tests(TestSquare, globals(), except_for='cpu')
if __name__ == '__main__': 
    run_tests() 