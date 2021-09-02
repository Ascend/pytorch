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


class TestConstantPadNdBackward(TestCase):  
    def op_exec_cpu(self, input1, pad_shape):
        input1.requires_grad = True
        output = torch.constant_pad_nd(input1, pad_shape)
        output.backward(torch.ones_like(output))
        input_grad = input1.grad
        output = output.detach()
        return output, input_grad

    def op_exec_npu(self, input1, pad_shape):
        input1.requires_grad = True
        output = torch.constant_pad_nd(input1, pad_shape)
        output.backward(torch.ones_like(output))
        input_grad = input1.grad
        output = output.detach().cpu()
        input_grad = input_grad.cpu()
        return output, input_grad
        
    def constant_pad_nd_backward_shape_format(self, shape_format):
        for item in shape_format:
            input_cpu, input_npu = create_common_tensor(item[0], 1, 1)
            pad_shape = item[1]
            if input_cpu.dtype == torch.float16:
                cpu_output, cpu_input_grad = self.op_exec_cpu(input_cpu.float(), pad_shape)
                cpu_output = cpu_output.half()
                cpu_input_grad = cpu_input_grad.half()
            else:
                cpu_output, cpu_input_grad = self.op_exec_cpu(input_cpu, pad_shape)
            npu_output, npu_input_grad = self.op_exec_npu(input_npu, pad_shape) 
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_input_grad, npu_input_grad)
        
    def test_constant_pad_nd_backward_shape_1d(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        pad_list = [(1, 2)]
        shape_list = [(16)]
        shape_format = [
            [[i, j, k], l]  for i in dtype_list for j in format_list for k in shape_list for l in pad_list            
        ]
        self.constant_pad_nd_backward_shape_format(shape_format)
        
    def test_constant_pad_nd_backward_shape_2d(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        pad_list = [(1, 2), 
                    (1, 2, 2, 2)]
        shape_list = [(2, 16)]
        shape_format = [
            [[i, j, k], l]  for i in dtype_list for j in format_list for k in shape_list for l in pad_list            
        ]
        self.constant_pad_nd_backward_shape_format(shape_format)
    
    def test_constant_pad_nd_backward_shape_3d(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        pad_list = [(1, 2), 
                    (1, 2, 2, 2), 
                    (1, 2, 2, 2, 2, 2)]
        shape_list = [(2, 16, 5)]
        shape_format = [
            [[i, j, k], l]  for i in dtype_list for j in format_list for k in shape_list for l in pad_list            
        ]
        self.constant_pad_nd_backward_shape_format(shape_format)
    
    def test_constant_pad_nd_backward_shape_4d(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        pad_list = [(1, 2), 
                    (1, 2, 2, 2), 
                    (1, 2, 2, 2, 2, 2), 
                    (1, 2, 2, 2, 2, 2, 2, 2)]
        shape_list = [(2, 16, 5, 8)]
        shape_format = [
            [[i, j, k], l]  for i in dtype_list for j in format_list for k in shape_list for l in pad_list            
        ]
        self.constant_pad_nd_backward_shape_format(shape_format)

    def test_constant_pad_nd_backward_shape_5d(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        pad_list = [(1, 2), 
                    (1, 2, 2, 2), 
                    (1, 2, 2, 2, 2, 2), 
                    (1, 2, 2, 2, 2, 2, 2, 2),
                    (1, 2, 2, 2, 2, 2, 2, 2, 2, 2)]
        shape_list = [(2, 16, 5, 8, 3)]
        shape_format = [
            [[i, j, k], l]  for i in dtype_list for j in format_list for k in shape_list for l in pad_list            
        ]
        self.constant_pad_nd_backward_shape_format(shape_format)


instantiate_device_type_tests(TestConstantPadNdBackward, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
