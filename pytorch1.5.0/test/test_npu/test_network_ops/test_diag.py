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

class TestDiag(TestCase):
    def cpu_op_exec(self, input, diagonal):
        output = torch.diag(input, diagonal=diagonal)
        output = output.numpy()
        return output

    def npu_op_exec(self, input, diagonal):
        output = torch.diag(input, diagonal=diagonal)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_out(self, input, diagonal, out):
        torch.diag(input, diagonal=diagonal, out=out)
        output = out.numpy()
        return output

    def npu_op_exec_out(self, input, diagonal, out):
        torch.diag(input, diagonal=diagonal, out=out)
        output = out.to("cpu")
        output = output.numpy()
        return output

    def generate_npu_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        output1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        npu_output1 = torch.from_numpy(output1)
        return npu_input1, npu_output1

    def test_diag_common_shape_format(self, device):
        shape_format = [
            [[np.float32, -1, [16]], 0],    # test the condition of 1-dimension
            [[np.float32, -1, [1024]], 0],    
            [[np.float32, -1, [5, 5]], 0],  # test the condition of 2-dimension
            [[np.float32, -1, [256, 256]], 0],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input, item[1])
            npu_output = self.npu_op_exec(npu_input, item[1])
            self.assertRtolEqual(cpu_output, npu_output)
    
    def test_diag_float32_out(self, device):
        shape_format = [
            [[np.float32, -1, [16]], [np.float32, -1, [20]], 0],    # test the condition of 1-dimension
            [[np.float32, -1, [1024]], [np.float32, -1, [20, 20]], 0],    
            [[np.float32, -1, [5, 5]], [np.float32, -1, [5, 5, 5]], 0],  # test the condition of 2-dimension
            [[np.float32, -1, [256, 256]], [np.float32, -1, [256]], 0],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, item[2])
            npu_output = self.npu_op_exec_out(npu_input1, item[2], npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_diag_float16_out(self, device):
        shape_format = [
            [[np.float16, -1, [16]], [np.float16, -1, [20]], 0],    # test the condition of 1-dimension
            [[np.float16, -1, [1024]], [np.float16, -1, [20, 20]], 0],    
            [[np.float16, -1, [5, 5]], [np.float16, -1, [5, 5, 5]], 0],  # test the condition of 2-dimension
            [[np.float16, -1, [256, 256]], [np.float16, -1, [256]], 0],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, item[2])
            npu_output = self.npu_op_exec_out(npu_input1, item[2], npu_input2)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_diag_float16_shape_format(self, device):
        def cpu_op_exec_fp16(input, diagonal):
            input = input.to(torch.float32)
            output = torch.diag(input, diagonal)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        shape_format = [
                [[np.float16, -1, [4]], 0],     # test the condition of 1-dimension
                [[np.float16, -1, [512]], 0],
                [[np.float16, -1, [4, 4]], 0],  # test the condition of 2-dimension
                [[np.float16, -1, [256, 256]], 0],
        ] 
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_output = cpu_op_exec_fp16(cpu_input, item[1])
            npu_output = self.npu_op_exec(npu_input, item[1])
            self.assertRtolEqual(cpu_output, npu_output)  

instantiate_device_type_tests(TestDiag, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
        
