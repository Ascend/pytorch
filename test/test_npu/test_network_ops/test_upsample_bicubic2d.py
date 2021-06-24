# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION. 
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

import torch
import numpy as np
import sys
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestUpsampleBicubic2d(TestCase):

    def cpu_op_exec(self, input1, output_size, align_corners, scale_h, scale_w): 
        output = torch._C._nn.upsample_bicubic2d(input1, output_size, align_corners, scale_h, scale_w)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, output_size, align_corners, scale_h, scale_w): 
        output = torch._C._nn.upsample_bicubic2d(input1, output_size, align_corners, scale_h, scale_w)
        output = output.to("cpu") 
        output = output.numpy() 
        return output 

    # float32 [0.0002, 0.0001]
    #pylint: disable=unused-argument
    def test_upsample_bicubic2d_common_shape_format(self, device):
        shape_format = [
                        # same size
                        [[np.float32, -1, (1, 1, 1, 1)], (1, 1), True, 0, 0, 0, 255],  # case 1
                        [[np.float32, -1, (2, 65535, 2, 2)], (2, 2), True, 0, 0, 0, 255],  # case 2
                        [[np.float32, -1, (65535, 2, 4, 8)], (4, 8), True, 0, 0, 0, 255],  # case 3
                        [[np.float32, -1, (2, 4, 65535, 2)], (65535, 2), True, 0, 0, 0, 255],  # case 4
                        [[np.float32, -1, (1, 31, 149, 2)], (149, 2), True, 0, 0, 0, 255],  # case 5
                        [[np.float32, -1, (10, 10, 786432, 8)], (786432, 8), True, 0, 0, 0, 255],  # case 6
                        [[np.float32, -1, (32, 32, 32, 32)], (32, 32), True, 0, 0, 0, 3402823500.0],  # case 7
                        [[np.float32, -1, (10, 10, 786432, 8)], (786432, 8), False, 0, 0, 0, 255],  # case 8
                        [[np.float32, -1, (32, 32, 32, 32)], (32, 32), False, 0, 0, 0, 3402823500.0],  # case 9
                        
                        # align_corners = True
                        [[np.float32, -1, (1, 1, 1, 1)], (2, 2), True, 0, 0, 0, 255],  # case 10
                        [[np.float32, -1, (1, 1, 2, 2)], (4, 4), True, 0, 0, 0, 255],  # case 11
                        [[np.float32, -1, (2, 2, 1, 1)], (2, 2), True, 0, 0, 0, 255],  # case 12
                        [[np.float32, -1, (2, 2, 2, 2)], (10, 10), True, 0, 0, 0, 255],  # case 13
                        [[np.float32, -1, (1, 31, 149, 2)], (2, 149), True, 0, 0, 0, 255],  # case 14
                        #[[np.float32, -1, (32, 32, 32, 32)], (64, 64), True, 0, 0, 0, 255],  # case 15
                        #[[np.float32, -1, (32, 32, 32, 32)], (64, 64), True, 0, 0, 0, 3402823500.0],  # case 16
                        
                        # align_corners = False
                        [[np.float32, -1, (1, 1, 1, 1)], (2, 2), False, 0.5, 0.5, 0, 255],  # case 17
                        [[np.float32, -1, (1, 1, 2, 2)], (4, 4), False, 0.5, 0.5, 0, 255],  # case 18
                        [[np.float32, -1, (2, 2, 1, 1)], (2, 2), False, 0.5, 0.5, 0, 255],  # case 19
                        [[np.float32, -1, (2, 2, 2, 2)], (10, 10), False, 0.5, 0.5, 0, 255],  # case 20
                        [[np.float32, -1, (1, 31, 149, 2)], (2, 149), False, 0.5, 0.5, 0, 255],  # case 21
                        [[np.float32, -1, (32, 32, 32, 32)], (64, 64), False, 0.5, 0.5, 0, 255],  # case 22
                        [[np.float32, -1, (32, 32, 32, 32)], (64, 64), False, 0.5, 0.5, 0, 3402823500.0]  # case 23
                       ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], item[5], item[6])
            cpu_output = self.cpu_op_exec(cpu_input1, item[1], item[2], item[3], item[4])
            npu_output = self.npu_op_exec(npu_input1, item[1], item[2], item[3], item[4])
            self.assertRtolEqual(cpu_output, npu_output)
  
    # float16 [0.002, 0.001]
    #pylint: disable=unused-argument
    def test_upsample_bicubic2d_float16_shape_format(self, device):
        def cpu_op_exec_fp16(input1, output_size, align_corners, scale_h, scale_w):
            input1 = input1.to(torch.float32)
            output = torch._C._nn.upsample_bicubic2d(input1, output_size, align_corners, scale_h, scale_w)
            output = output.numpy()
            output = output.astype(np.float16)
            return output
    
        shape_format = [
                        # same size
                        [[np.float16, -1, (1, 1, 1, 1)], (1, 1), True, 0, 0, 0, 255],  # case 24
                        [[np.float16, -1, (2, 65535, 2, 2)], (2, 2), True, 0, 0, 0, 255],  # case 25
                        [[np.float16, -1, (65535, 2, 4, 8)], (4, 8), True, 0, 0, 0, 255],  # case 26
                        [[np.float16, -1, (2, 4, 65535, 2)], (65535, 2), True, 0, 0, 0, 255],  # case 27
                        [[np.float16, -1, (1, 31, 149, 2)], (149, 2), True, 0, 0, 0, 255],  # case 28
                        [[np.float16, -1, (10, 10, 786432, 8)], (786432, 8), True, 0, 0, 0, 255],  # case 29
                        [[np.float16, -1, (32, 32, 32, 32)], (32, 32), True, 0, 0, 0, 6550.0],  # case 30
                        [[np.float16, -1, (10, 10, 786432, 8)], (786432, 8), False, 0, 0, 0, 255],  # case 31
                        [[np.float16, -1, (32, 32, 32, 32)], (32, 32), False, 0, 0, 0, 6550.0],  # case 32
    
                        # align_corners = True
                        [[np.float16, -1, (1, 1, 1, 1)], (2, 2), True, 0, 0, 0, 255],  # case 33
                        [[np.float16, -1, (1, 1, 2, 2)], (4, 4), True, 0, 0, 0, 255],  # case 34
                        [[np.float16, -1, (2, 2, 1, 1)], (2, 2), True, 0, 0, 0, 255],  # case 35
                        [[np.float16, -1, (2, 2, 2, 2)], (10, 10), True, 0, 0, 0, 255],  # case 36
                        [[np.float16, -1, (1, 31, 149, 2)], (2, 149), True, 0, 0, 0, 255],  # case 37
                        [[np.float16, -1, (32, 32, 32, 32)], (64, 64), True, 0, 0, 0, 255],  # case 38
                        [[np.float16, -1, (32, 32, 32, 32)], (64, 64), True, 0, 0, 0, 6550.0],  # case 39
                        
                        # align_corners = False
                        [[np.float16, -1, (1, 1, 1, 1)], (2, 2), False, 0.5, 0.5, 0, 255],  # case 40
                        [[np.float16, -1, (1, 1, 2, 2)], (4, 4), False, 0.5, 0.5, 0, 255],  # case 41
                        [[np.float16, -1, (2, 2, 1, 1)], (2, 2), False, 0.5, 0.5, 0, 255],  # case 42
                        [[np.float16, -1, (2, 2, 2, 2)], (10, 10), False, 0.5, 0.5, 0, 255],  # case 43
                        [[np.float16, -1, (1, 31, 149, 2)], (2, 149), False, 0.5, 0.5, 0, 255],  # case 44
                        [[np.float16, -1, (32, 32, 32, 32)], (64, 64), False, 0.5, 0.5, 0, 255],  # case 45
                        [[np.float16, -1, (32, 32, 32, 32)], (64, 64), False, 0.5, 0.5, 0, 6550.0]  # case 46
                       ]
        
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], item[5], item[6])
            cpu_output = cpu_op_exec_fp16(cpu_input1, item[1], item[2], item[3], item[4])
            npu_output = self.npu_op_exec(npu_input1, item[1], item[2], item[3], item[4])
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestUpsampleBicubic2d, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()