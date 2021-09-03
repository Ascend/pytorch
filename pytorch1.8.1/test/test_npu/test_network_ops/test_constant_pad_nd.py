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

class TestConstantPadNd(TestCase):  
    def op_exec_cpu(self, input1, pad_shape):
        output = torch.constant_pad_nd(input1, pad_shape)
        output = output.numpy()        
        return output

    def op_exec_npu(self, input1, pad_shape):
        input1 = input1.to("npu")
        output = torch.constant_pad_nd(input1, pad_shape)
        output = output.to("cpu")
        output = output.numpy()
        return output
        
    def constant_pad_nd_shape_format(self, shape_format):
        for item in shape_format:
          print(item)
          input_cpu, input_npu = create_common_tensor(item[0], 1, 1)
          pad_shape = item[1]
          if input_cpu.dtype == torch.float16:
              input_cpu = input_cpu.to(torch.float32)
              input_npu = input_npu.to(torch.float32)
          cpu_output = self.op_exec_cpu(input_cpu, pad_shape)
          npu_output = self.op_exec_npu(input_npu, pad_shape) 
          cpu_output = cpu_output.astype(npu_output.dtype)            
          self.assertRtolEqual(cpu_output, npu_output)        
        
    def test_constant_pad_nd_shape_1d(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [0, 3]
        pad_list = [(1,2)]
        shape_format = [
            [[i, j, [18]], k]  for i in dtype_list for j in format_list for k in pad_list            
        ]
        
        self.constant_pad_nd_shape_format(shape_format)
        
    def test_constant_pad_nd_shape_nd(self, device):
        dtype_list = [np.float16, np.float32]
        format_list = [0, 3]
        pad_list = [(1,2,2,2),(1,2)]
        shape_list = [(16,128), (2,16,128), (1,2,16,128)]
        shape_format = [
            [[i, j, k], l]  for i in dtype_list for j in format_list for k in shape_list for l in pad_list            
        ]
        
        self.constant_pad_nd_shape_format(shape_format)

    def test_constant_pad_nd_shape_nd_int32(self, device):
        dtype_list = [np.int32]
        format_list = [0]
        pad_list = [(1,2,2,2),(1,2)]
        shape_list = [(16,128), (2,16,128), (1,2,16,128)]
        shape_format = [
            [[i, j, k], l]  for i in dtype_list for j in format_list for k in shape_list for l in pad_list            
        ]
        
        self.constant_pad_nd_shape_format(shape_format)        

instantiate_device_type_tests(TestConstantPadNd, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
