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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


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
    
    def create_shape_format32(self):
        dtype_list = [np.float32]
        format_list = [-1]
        shape_list = [(1, 1, 1, 1), (1, 31, 149, 2), (32, 32, 32, 32), (65535, 2, 4, 8)]
        size_list = [(2, 2), (4, 4)]
        align_corners_list = [True, False]
        scale_h = [0, 0.5]
        scale_w = [0, 0.5]

        shape_format = [[[i, j, k], h, f, e, g] for i in dtype_list
                        for j in format_list for k in shape_list 
                        for h in size_list for f in align_corners_list
                        for e in scale_h for g in scale_w]
        
        return shape_format

    def create_shape_format16(self):
        dtype_list1 = [np.float16]
        format_list1 = [-1]
        shape_list1 = [(1, 1, 1, 1), (1, 31, 149, 2), (32, 32, 32, 32), (65535, 2, 4, 8)]
        size_list1 = [(2, 2), (4, 4)]
        align_corners_list1 = [True, False]
        scale_h1 = [0, 0.5]
        scale_w1 = [0, 0.5]

        shape_format1 = [[[i, j, k], h, f, e, g] for i in dtype_list1
                        for j in format_list1 for k in shape_list1
                        for h in size_list1 for f in align_corners_list1
                        for e in scale_h1 for g in scale_w1]
        
        return shape_format1

    def test_upsample_bicubic2d_common_shape_format(self):
        for item in self.create_shape_format32():
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 255)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1], item[2], item[3], item[4])
            npu_output = self.npu_op_exec(npu_input1, item[1], item[2], item[3], item[4])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_upsample_bicubic2d_float16_shape_format(self):
        def cpu_op_exec_fp16(input1, output_size, align_corners, scale_h, scale_w):
            input1 = input1.to(torch.float32)
            output = torch._C._nn.upsample_bicubic2d(input1, output_size, align_corners, scale_h, scale_w)
            output = output.numpy()
            output = output.astype(np.float16)
            return output
    
        for item in self.create_shape_format16():
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 255)
            cpu_output = cpu_op_exec_fp16(cpu_input1, item[1], item[2], item[3], item[4])
            npu_output = self.npu_op_exec(npu_input1, item[1], item[2], item[3], item[4])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()