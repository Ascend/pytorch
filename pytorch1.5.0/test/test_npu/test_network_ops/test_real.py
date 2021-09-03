# Copyright (c) 2020 Huawei Technologies Co., Ltd
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
import random
import copy
from torch.autograd import Variable
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestReal(TestCase):
    def generate_data(self, min, max, shape, dtype):
        input1 = np.random.uniform(min, max, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        return npu_input1
        
    def cpu_op_exec(self, input1):
        output = torch.real(input1) 
        print(torch.real(input1))
        return output

    def npu_op_exec(self, input1):
        input1 = input1.to("npu")
        output = torch.real(input1) 
        output = output.to("cpu")
        return output 
        
    def test_real_float32_1(self, device):
        npu_input1 = self.generate_data(0, 100, (4, ), np.float32)
        cpu_output = self.npu_op_exec(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)
        
    def test_real_float32_2(self, device):
        npu_input1 = self.generate_data(0, 100, (5, 1), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)
        
    def test_real_int32_1(self, device):
        npu_input1 = self.generate_data(0, 100, (4, ), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)
         
    def test_real_float32_1_1(self, device):
        npu_input1 = self.generate_data(0, 100, (5, 1, 1), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_real_float32_2_2(self, device):
        npu_input1 = self.generate_data(0, 100, (5, 1, 1), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1)
        npu_output = self.npu_op_exec(npu_input1)
        self.assertRtolEqual(cpu_output, npu_output)
 

instantiate_device_type_tests(TestReal, globals(), except_for='cpu')
if __name__ == '__main__':
    run_tests()