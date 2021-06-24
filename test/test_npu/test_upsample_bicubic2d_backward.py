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


class TestResizeGradD(TestCase):

    def generate_grads_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1) 
        return npu_input1

    def cpu_op_exec(self, grads, shape_x, output_size, align_corners, scale_h, scale_w):
        input1 = torch.ones(shape_x) 
        flag = 0
        if input1.dtype != torch.float32:
            input1 = input1.to(torch.float32)
            flag = 1
        input_data = input1.clone().detach().requires_grad_(True)                                                                              
        y = torch._C._nn.upsample_bicubic2d(input_data, output_size, align_corners, scale_h, scale_w)
        y.backward(grads)
        output = input_data.grad
        if flag == 1:
            output = output.to(torch.float16)
        output = output.numpy()
        return output 
     
    
    def npu_op_exec(self, grads, shape_x, output_size, align_corners, scale_h, scale_w): 
        input1 = torch.ones(shape_x)
        input1 = input1.to("npu")
        grads = grads.to("npu")
        input_data = input1.clone().detach().requires_grad_(True) 
        y = torch._C._nn.upsample_bicubic2d(input_data, output_size, align_corners, scale_h, scale_w)
        y.backward(grads)
        output = input_data.grad
        output = output.to("cpu") 
        output = output.numpy() 
        return output   
    
    #pylint: disable=too-many-arguments
    def resize_grad_d(self, shape_x, output_size, scales, align_corners, minVal, maxVal, dtype):
        grads = self.generate_grads_data(minVal, maxVal, (shape_x[0], shape_x[1], output_size[0], output_size[1]), dtype)
        scale_h = scales[0]
        scale_w = scales[1]
        cpu_output = self.cpu_op_exec(grads, shape_x, output_size, align_corners, scale_h, scale_w)
        npu_output = self.npu_op_exec(grads, shape_x, output_size, align_corners, scale_h, scale_w)
        self.assertRtolEqual(cpu_output, npu_output)
    
    #pylint: disable=unused-argument
    def test_resize_grad_d(self, device):
        testcases = \
        [  
          # special case : same size fp32
          [[4, 3, 128, 64], [128, 64], [0, 0], True, -3.4028235E-14, 3.4028235E-14, np.float32], # case 1
          [[128, 3, 128, 64], [128, 64], [0, 0], False, -3.4028235E14, 3.4028235E14, np.float32], # case 2
          [[65535, 2, 4, 8], [4, 8], [0, 0], True, -10, 10, np.float32], # case 3
          [[2, 65535, 4, 8], [4, 8], [0, 0], True, -10, 10, np.float32], # case 4
          [[2, 4, 65535, 8], [65535, 8], [0, 0], True, -10, 10, np.float32], # case 5
          [[2, 4, 8, 65535], [8, 65535], [0, 0], True, -10, 10, np.float32], # case 6
          [[2, 4, 8, 786432], [8, 786432], [0, 0], True, -10, 10, np.float32], # case 7        
          
          # special case : same size fp16
          [[4, 3, 128, 64], [128, 64], [0, 0], True, -3.4028235E-6, 3.4028235E-6, np.float16], # case 8
          [[128, 3, 128, 64], [128, 64], [0, 0], False, -3.4028235E3, 3.4028235E4, np.float16], # case 9
          [[65535, 2, 4, 8], [4, 8], [0, 0], True, -10, 10, np.float16], # case 10
          [[2, 65535, 4, 8], [4, 8], [0, 0], True, -10, 10, np.float16], # case 11
          [[2, 4, 65535, 8], [65535, 8], [0, 0], True, -10, 10, np.float16], # case 12
          [[2, 4, 8, 65535], [8, 65535], [0, 0], True, -10, 10, np.float16], # case 13
          [[2, 4, 8, 786432], [8, 786432], [0, 0], True, -10, 10, np.float16], # case 14
          
          # common case fp32
          [[4, 3, 128, 64], [128, 128], [0, 0], True, -3.4028235E-14, 3.4028235E-14, np.float32], # case 15
          [[128, 3, 128, 64], [128, 128], [0, 0], False, -3.4028235E14, 3.4028235E14, np.float32], # case 16
          [[65535, 2, 4, 8], [16, 32], [0, 0], True, -10, 10, np.float32], # case 17
          [[2, 65535, 4, 8], [8, 16], [0, 0], True, -10, 10, np.float32], # case 18
          [[2, 4, 65535, 8], [65535, 16], [0, 0], False, -10, 10, np.float32], # case 19
          [[2, 4, 8, 65535], [16, 65535], [0, 0], True, -10, 10, np.float32], # case 20
          [[2, 4, 8, 786432], [16, 786432], [0, 0], True, -10, 10, np.float32], # case 21
          
          # common case fp16
          [[4, 3, 128, 64], [128, 128], [0, 0], False, -3.4028235E-6, 3.4028235E-5, np.float16], # case 22
          [[128, 3, 128, 64], [128, 128], [0, 0], True, -3.4028235E3, 3.4028235E3, np.float16], # case 23
          [[65535, 2, 4, 8], [16, 32], [0, 0], True, -10, 10, np.float16], # case 24
          [[2, 65535, 4, 8], [8, 16], [0, 0], True, -10, 10, np.float16], # case 25
          [[2, 4, 65535, 8], [65535, 16], [0, 0], False, -10, 10, np.float16], # case 26
          [[2, 4, 8, 65535], [16, 65535], [0, 0], True, -10, 10, np.float16], # case 27
          [[2, 4, 8, 786432], [16, 786432], [0, 0], True, -10, 10, np.float16] # case 28
          
        ]
        case = 1 
        for item in testcases: 
            print("==========\nrunning case:{}...".format(case))
            self.resize_grad_d(item[0], item[1], item[2], item[3], item[4], item[5], item[6])
            print("case:{} cmp success\n".format(case))
            case += 1


instantiate_device_type_tests(TestResizeGradD, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:1")
    run_tests()
