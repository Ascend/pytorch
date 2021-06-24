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

class TestAcos(TestCase):
    def cpu_op_exec(self, input):
        output = torch.acos(input) 
        output = output.numpy()
        return output
 
    def npu_op_exec(self, input):
        output = torch.acos(input) 
        output = output.to("cpu") 
        output = output.numpy()
        return output  
        
    def test_acos_common_shape_format(self, device):
        shape_format = [
                [[np.float32, -1, 1]],
                [[np.float32, -1, (64, 10)]], 
                [[np.float32, -1, (32, 1, 3)]]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], -1, 1)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_acos_float16_shape_format(self, device):
        def cpu_op_exec_fp16(input):
            input = input.to(torch.float32)
            output = torch.acos(input)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        shape_format = [
                [[np.float16, -1, 1]], 
                [[np.float16, -1, (64, 10)]],    
                [[np.float16, -1, (31, 1, 3)]]
        ] 

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], -1, 1)
            cpu_output = cpu_op_exec_fp16(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)  

instantiate_device_type_tests(TestAcos, globals(), except_for="cpu")
if __name__ == "__main__":
    torch.npu.set_device("npu:5")
    run_tests()
        
