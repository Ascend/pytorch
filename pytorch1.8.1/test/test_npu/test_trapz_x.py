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

class TestTrapzX(TestCase):
    def generate_data(self, min, max, shape, dtype): 
        input1 = np.random.uniform(min, max, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1) 
        return npu_input1

    def cpu_op_exec(self, input1, input2, dim):
        output = torch.trapz(input1, input2, dim=dim)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2, dim = -1):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        output = torch.trapz(input1, input2, dim=dim)
        output = output.to("cpu")
        output = output.numpy()
        return output
        
    def cpu_op_exec_float16(self, input1, input2): 
        input1 = input1.to(torch.float32)
        input2 = input2.to(torch.float32)
        output = torch.trapz(input1, input2)
        output = output.numpy()
        output = output.astype(np.float16)
        return output 

    def cpu_op_exec_trapz_dx(self, input1, dx, dim):
        output = torch.trapz(input1, dx=dx, dim=dim)
        output = output.numpy()
        return output

    def npu_op_exec_trapz_dx(self, input1, dx, dim):
        output = torch.trapz(input1, dx=dx, dim=dim)
        output = output.to("cpu")
        output = output.numpy()
        return output  

    def test_trapz_x(self, device):
        shape_format = [
            [[np.float32, -1, (2,3)]],
            [[np.float32, -1, (2,2,3)]],
            [[np.float32, -1, (7,2,4,5)]]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 0, 100)
            cpu_output1 = self.cpu_op_exec(cpu_input1, cpu_input2, -1)
            npu_output1 = self.npu_op_exec(npu_input1, npu_input2, -1)
            cpu_output2 = self.cpu_op_exec(cpu_input1, cpu_input2, 1)
            npu_output2 = self.npu_op_exec(npu_input1, npu_input2, 1)
            cpu_output3 = self.cpu_op_exec_trapz_dx(cpu_input1,2,1)
            npu_output3 = self.npu_op_exec_trapz_dx(npu_input1,2,1)
            self.assertRtolEqual(cpu_output1, npu_output1)
            self.assertRtolEqual(cpu_output2, npu_output2)
            self.assertRtolEqual(cpu_output3, npu_output3)

    def test_trapz_x_float16(self, device):
        cpu_input1 = self.generate_data(0, 100, (2,2,3), np.float16)
        cpu_input2 = self.generate_data(0, 100, (2,2,3), np.float16)
        cpu_output = self.cpu_op_exec_float16(cpu_input1, cpu_input2)
        npu_output = self.npu_op_exec(cpu_input1, cpu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestTrapzX, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:2")
    run_tests()
