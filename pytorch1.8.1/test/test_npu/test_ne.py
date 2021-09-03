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
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestNe(TestCase):

    def cpu_op_exec_scalar(self, input1, other):
        output = torch.ne(input1, other)
        output = output.numpy()
        return output

    def npu_op_exec_scalar(self,input1, other):
        output = torch.ne(input1, other)
        output1 = output.to("cpu")
        output2 = output1.numpy()
        return output2

    def cpu_op_exec(self, input1, other):
        output = torch.ne(input1, other)
        output = output.numpy()
        return output

    def npu_op_exec(self,input1, other):
        output = torch.ne(input1, other)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_(self,input1, other):
        torch.ne_(input1,other)
        output = input1.numpy()
        return output

    def npu_op_exec_(self,input1, other):
        torch.ne_(input1, other)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_scalar_(self,input1, other):
        torch.ne_(input1,other)
        output = input1.numpy()
        return output

    def npu_op_exec_scalar_(self,input1, other):
        torch.ne_(input1, other)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_scalar_out(self,input1,other, out):
        torch.ne(input1,other, out=out)
        output = out.numpy()
        return output

    def npu_op_exec_scalar_out(self,input1, other, out):
        torch.ne(input1, other, out=out)
        output = out.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_out(self,input1,other, out):
        torch.ne(input1,other, out=out)
        output = out.numpy()
        return output

    def npu_op_exec_out(self,input1, other, out):
        torch.ne(input1, other, out=out)
        output = out.to("cpu")
        output = output.numpy()
        return output

    def test_ne_scalar_common_shape_format(self, device):
        shape_format = [
                [[np.float32,0 , (2,4, 3)], 3],
                [[np.float32, 3, (2, 3)], 2],
                [[np.float32, 0, (3, 2)], 8],
                [[np.int8, 0 , (4, 3)],3],
                [[np.uint8, -1, (2,4, 3)],3],
                [[np.int32, 0, (2, 6)],6]
                ]
        for item in shape_format:            
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 10)
            cpu_output = self.cpu_op_exec_scalar(cpu_input1, item[1])
            npu_output = self.npu_op_exec_scalar(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)       

    def test_ne_common_shape_format(self, device):
        shape_format = [
                [[np.float32,0 , (2, 4, 3)], [np.float32,0 , (2, 4, 3)]],
                [[np.float32, 3, (2, 3)], [np.float32, 3, (2, 3)]],
                [[np.float32, 0, (3, 2)], [np.float32, 0, (3, 2)]],
                [[np.int8, 0 , (4, 3)], [np.int8, 0 , (4, 3)]],
                [[np.uint8, -1, (2,4, 3)], [np.uint8, -1, (2,4, 3)]],
                [[np.int32, 0, (2, 6)], [np.int32, 0, (2, 6)]],
                ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 1, 10)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)
    
    def test_ne_scalar_out_common_shape_format(self, device):
        shape_format = [
                [[np.float32,0 , (2, 4, 3)], 2, [np.bool, 0 , (2, 4, 3)]],
                [[np.float32, 3, (2, 3)],    3, [np.bool, -1, (2, 3)]],
                [[np.float32, 0, (3, 2)],    4, [np.bool, 0, (3, 2)]],

                [[np.int8, 0 , (4, 3)],      5, [np.bool, 0 , (4, 3)]],
                [[np.uint8, -1, (2,4, 3)],   6, [np.bool, -1, (2,4, 3)]],
                [[np.int32, 0, (2, 6)],      7, [np.bool, 0, (2, 6)]]
                ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 10)
            cpu_out, npu_out = create_common_tensor(item[2], 1, 10)
            cpu_output = self.cpu_op_exec_scalar_out(cpu_input1, item[1], cpu_out)
            npu_output = self.npu_op_exec_scalar_out(npu_input1, item[1], npu_out)
            self.assertRtolEqual(cpu_output, npu_output)
    
    def test_ne_out_common_shape_format(self, device):
        shape_format = [
                [[np.float32,0 , (2, 4, 3)], [np.float32,0 , (2, 4, 3)], [np.bool, 0 , (2, 4, 3)]],
                [[np.float32, 3, (2, 3)],    [np.float32, 3, (2, 3)],    [np.bool, -1, (2, 3)]],
                [[np.float32, 0, (3, 2)],    [np.float32, 0, (3, 2)],    [np.bool, 0, (3, 2)]],

                [[np.int8, 0 , (4, 3)],      [np.int8, 0 , (4, 3)],      [np.bool, 0 , (4, 3)]],
                [[np.uint8, -1, (2,4, 3)],   [np.uint8, -1, (2,4, 3)],   [np.bool, -1, (2,4, 3)]],
                [[np.int32, 0, (2, 6)],      [np.int32, 0, (2, 6)],      [np.bool, 0, (2, 6)]]
                ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 1, 10)
            cpu_out, npu_out = create_common_tensor(item[2], 1, 10)
            cpu_output = self.cpu_op_exec_out(cpu_input1, cpu_input2, cpu_out)
            npu_output = self.npu_op_exec_out(npu_input1, npu_input2, npu_out)
            self.assertRtolEqual(cpu_output, npu_output)
    
instantiate_device_type_tests(TestNe, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
