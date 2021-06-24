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

class test_bilinear(TestCase):
    def cpu_op_exec(self, input1, input2, weight, bias):
        if input1.dtype == torch.float16:
          input1 = input1.to(torch.float32)
          input2 = input2.to(torch.float32)
        outputs = torch.nn.functional.bilinear(input1, input2, weight, bias)
        outputs = outputs.detach().numpy()
        return outputs

    def npu_op_exec(self, input1, input2, weight, bias):
        outputs = torch.nn.functional.bilinear(input1, input2, weight, bias)
        outputs = outputs.cpu().detach().numpy()
        return outputs

    def test_add_common_shape_format1(self, device):
        shape_format = [  
                  [[np.float32, -1, (10,30)], [np.float32, -1, (10, 40)], [np.float32, -1, (5, 30, 40)],
                    [np.float32, -1, (5,)]],
                  [[np.float32, -1, (100, 30)], [np.float32, -1, (100, 40)], [np.float32, -1, (5, 30, 40)],
                    [np.float32, -1, (5,)]],
                  [[np.float32, -1, (100, 30)], [np.float32, -1, (100, 40)], [np.float32, -1, (5, 30, 40)],],
                  [[np.float32, -1, (10, 30, 40, 30)], [np.float32, -1, (10, 30, 40, 30)], 
                    [np.float32, -1, (30, 30, 30)],
                      [np.float32, -1, (30,)]],
                  [[np.float32, -1, (100,3)], [np.float32, -1, (1000, 4)], [np.float32, -1, (5, 3, 4)],
                    [np.float32, -1, (5,)]],
                  [[np.float16, -1, (2, 1, 1, 1)], [np.float16, -1, (2, 1, 1, 1)], [np.float16, -1, (5, 1, 1)],
                    [np.float16, -1, (5,)]],
                  [[np.float16, -1, (2, 50)], [np.float16, -1, (2, 50)], [np.float16, -1, (5, 50, 50)],
                    [np.float16, -1, (2, 4)]],
                  [[np.float16, -1, (2, 3)], [np.float16, -1, (2, 4)], [np.float16, -1, (2, 3, 4)],],
                  [[np.float16, -1, (2, 3)], [np.float16, -1, (2, 4)], [np.float16, -1, (4, 3, 4)],
                  [np.float16, -1, (4,)]],
                ]
        for item in shape_format:
            bias = [None, None]
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 1)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 1)
            cpu_input3, npu_input3 = create_common_tensor(item[2], 0, 1)
            if len(item)>3:
              cpu_input4, npu_input4 = create_common_tensor(item[3], 0, 1)
              bias = [cpu_input4, npu_input4]
            cpu_outputs = self.cpu_op_exec(cpu_input1, cpu_input2, cpu_input3, bias[0])
            npu_outputs = self.npu_op_exec(npu_input1, npu_input2, npu_input3, bias[1])
            self.assertRtolEqual(cpu_outputs, npu_outputs)
    
    def test_add_common_shape_format2(self, device):
        shape_format = [  
                  [[np.int32, -1, (10,30)], [np.int32, -1, (10, 40)], [np.int32, -1, (5, 30, 40)],
                    [np.int32, -1, (5,)]],
                  [[np.int32, -1, (100,30)], [np.int32, -1, (100, 40)], [np.int32, -1, (50, 30, 40)],
                    [np.int32, -1, (50,)]],
                  [[np.int32, -1, (100,30)], [np.int32, -1, (100, 40)], [np.int32, -1, (50, 30, 40)],],
                  [[np.int32, -1, (1, 1, 1, 1)], [np.int32, -1, (1, 1, 1, 1)], [np.int32, -1, (1, 1, 1)],
                    [np.int32, -1, (1,)]]
                  ]
        for item in shape_format:
            bias = [None, None]
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 1)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 1)
            cpu_input3, npu_input3 = create_common_tensor(item[2], 0, 1)
            if len(item)>3:
              cpu_input4, npu_input4 = create_common_tensor(item[3], 0, 1)
              bias = [cpu_input4, npu_input4]
            cpu_outputs = self.cpu_op_exec(cpu_input1, cpu_input2, cpu_input3, bias[0])
            npu_outputs = self.npu_op_exec(npu_input1, npu_input2, npu_input3, bias[1])
            self.assertRtolEqual(cpu_outputs, npu_outputs)
       
    def test_add_common_shape_format3(self, device):
        shape_format = [  
                [[np.float32, 0, (10,30)], [np.float32, 0, (10, 40)], [np.float32, 0, (5, 30, 40)],
                  [np.float32, 0, (5,)]],
                [[np.float32, 0, (100, 30)], [np.float32, 0, (100, 40)], [np.float32, 0, (5, 30, 40)],
                  [np.float32, 0, (5,)]],
                [[np.float32, 0, (100, 30)], [np.float32, 0, (100, 40)], [np.float32, 0, (5, 30, 40)],],
                [[np.float32, 0, (10, 30, 40, 30)], [np.float32, 0, (10, 30, 40, 30)], 
                  [np.float32, 0, (30, 30, 30)],
                    [np.float32, 0, (30,)]],
                [[np.float32, 0, (100,3)], [np.float32, 0, (1000, 4)], [np.float32, 0, (5, 3, 4)],
                  [np.float32, 0, (5,)]],
                [[np.float16, 0, (2, 1, 1, 1)], [np.float16, 0, (2, 1, 1, 1)], [np.float16, 0, (5, 1, 1)],
                  [np.float16, 0, (5,)]],
                [[np.float16, 0, (2, 50)], [np.float16, 0, (2, 50)], [np.float16, 0, (5, 50, 50)],
                  [np.float16, 0, (2, 4)]],
                [[np.float16, 0, (2, 3)], [np.float16, 0, (2, 4)], [np.float16, 0, (2, 3, 4)],],
                [[np.float16, 0, (2, 3)], [np.float16, 0, (2, 4)], [np.float16, 0, (4, 3, 4)],
                [np.float16, 0, (4,)]],
              ]
        for item in shape_format:
          bias = [None, None]
          cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 1)
          cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 1)
          cpu_input3, npu_input3 = create_common_tensor(item[2], 0, 1)
          if len(item)>3:
            cpu_input4, npu_input4 = create_common_tensor(item[3], 0, 1)
            bias = [cpu_input4, npu_input4]
          cpu_outputs = self.cpu_op_exec(cpu_input1, cpu_input2, cpu_input3, bias[0])
          npu_outputs = self.npu_op_exec(npu_input1, npu_input2, npu_input3, bias[1])
          self.assertRtolEqual(cpu_outputs, npu_outputs)

    def test_add_common_shape_format4(self, device):
        shape_format = [  
                [[np.float32, 3, (10,30)], [np.float32, 3, (10, 40)], [np.float32, 3, (5, 30, 40)],
                  [np.float32, 3, (5,)]],
                [[np.float32, 3, (100, 30)], [np.float32, 3, (100, 40)], [np.float32, 3, (5, 30, 40)],
                  [np.float32, 3, (5,)]],
                [[np.float32, 3, (100, 30)], [np.float32, 3, (100, 40)], [np.float32, 3, (5, 30, 40)],],
                [[np.float32, 3, (10, 30, 40, 30)], [np.float32, 3, (10, 30, 40, 30)], 
                  [np.float32, 3, (30, 30, 30)],
                    [np.float32, 3, (30,)]],
                [[np.float32, 29, (100,3)], [np.float32, 29, (1000, 4)], [np.float32, 29, (5, 3, 4)],
                  [np.float32, 29, (5,)]],
                [[np.float16, 29, (2, 1, 1, 1)], [np.float16, 29, (2, 1, 1, 1)], [np.float16, 29, (5, 1, 1)],
                  [np.float16, 29, (5,)]],
                [[np.float16, 29, (2, 50)], [np.float16, 29, (2, 50)], [np.float16, 29, (5, 50, 50)],
                  [np.float16, 29, (2, 4)]],
                [[np.float16, 29, (2, 3)], [np.float16, 29, (2, 4)], [np.float16, 29, (2, 3, 4)],],
                [[np.float16, 29, (2, 3)], [np.float16, 29, (2, 4)], [np.float16, 29, (4, 3, 4)],
                [np.float16, 29, (4,)]],
              ]
        for item in shape_format:
          bias = [None, None]
          cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 1)
          cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 1)
          cpu_input3, npu_input3 = create_common_tensor(item[2], 0, 1)
          if len(item)>3:
            cpu_input4, npu_input4 = create_common_tensor(item[3], 0, 1)
            bias = [cpu_input4, npu_input4]
          cpu_outputs = self.cpu_op_exec(cpu_input1, cpu_input2, cpu_input3, bias[0])
          npu_outputs = self.npu_op_exec(npu_input1, npu_input2, npu_input3, bias[1])
          self.assertRtolEqual(cpu_outputs, npu_outputs)
          
instantiate_device_type_tests(test_bilinear, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:5")
    run_tests()
