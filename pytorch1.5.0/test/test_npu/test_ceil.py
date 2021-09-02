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
 
class TestCeil(TestCase):
    def cpu_op_exec(self, input):
        output = torch.ceil(input) 
        output = output.numpy()
        return output
 
    def npu_op_exec(self, input):
        output = torch.ceil(input) 
        output = output.to("cpu")
        output = output.numpy() 
        return output
    
    def cpu_op_out_exec(self, input):
        output = torch.ceil_(input) 
        output = output.numpy()
        return output
 
    def npu_op_out_exec(self, input):
        output = torch.ceil_(input) 
        output = output.to("cpu")
        output = output.numpy() 
        return output
        
    def test_ceil_common_shape_format(self, device):
        shape_format = [
                [[np.float32, 0, 1]],
                [[np.float32, 3, (64, 10)]],
                [[np.float32, 29, (256, 2048, 7, 7)]],
                [[np.float32, 3, 1]],
                [[np.float32, 0, (64, 10)]],
                [[np.float32, 3, (256, 2048, 7, 7)]],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_ceil_float16_shape_format(self, device):
        def cpu_op_exec_fp16(input):
            input = input.to(torch.float32)
            output = torch.ceil(input)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        shape_format = [
                [[np.float16, 0, 1]], 
                [[np.float16, 3, (64, 10)]],    
                [[np.float16, 29, (31, 1, 3)]],
                [[np.float16, 0, 1]], 
                [[np.float16, 3, (64, 10)]],    
                [[np.float16, 29, (31, 1, 3)]],
                [[np.float16, 0, 2,2]], 
                [[np.float16, 3, (64, 10, 24)]],    
                [[np.float16, 29, (31, 1, 3, 22)]]
        ] 
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_output = cpu_op_exec_fp16(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_ceil_out_common_shape_format(self, device):
        shape_format = [
                [[np.float32, 0, 1]],
                [[np.float32, 3, (64, 10)]],
                [[np.float32, 29, (256, 2048, 7, 7)]],
                [[np.float32, 3, 1]],
                [[np.float32, 0, (64, 10)]],
                [[np.float32, 3, (256, 2048, 7, 7)]],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_out_exec(cpu_input)
            npu_output = self.npu_op_out_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_ceil_out_float16_shape_format(self, device):
        def cpu_op_exec_fp16(input):
            input = input.to(torch.float32)
            output = torch.ceil_(input)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        shape_format = [
                [[np.float16, 0, 1]], 
                [[np.float16, 3, (64, 10)]],    
                [[np.float16, 29, (31, 1, 3)]],
                [[np.float16, 3, (64, 10)]],    
                [[np.float16, 29, (31, 1, 3)]],
                [[np.float16, 0, 2,2]], 
                [[np.float16, 3, (64, 10, 24)]],    
                [[np.float16, 29, (31, 1, 3, 22)]]
        ] 
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_output = cpu_op_exec_fp16(cpu_input)
            npu_output = self.npu_op_out_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)    

instantiate_device_type_tests(TestCeil, globals(), except_for="cpu") 
if __name__ == "__main__":
    run_tests()