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
import random
import copy
from torch.autograd import Variable
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestSolve(TestCase):
    def generate_data(self, min, max, shape, dtype):
        input = np.random.uniform(min, max, shape).astype(dtype)
        npu_input = torch.from_numpy(input)
        return npu_input
        
    def cpu_op_exec(self, input1, input2):
        X, LU = torch.solve(input2, input1) 
        return X

    def npu_op_exec(self, input1, input2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        X, LU = torch.solve(input2, input1) 
        X = X.to("cpu")
        return X 

    def test_solve_float16_2(self, device):
        def cpu_op_exec_float16_2(input1, input2):
            input1 = input1.to(torch.float32)
            input2 = input2.to(torch.float32)
            X, LU = torch.solve(input2, input1)
            X = X.numpy()
            X = X.astype(np.float16)
            return X
        npu_input1 = self.generate_data(0, 100, (2, 2), np.float16)
        npu_input2 = self.generate_data(0, 100, (2, 1), np.float16)
        cpu_output = cpu_op_exec_float16_2(npu_input1, npu_input2)
        # npu_output = self.npu_op_exec(npu_input1, npu_input2)
        #self.assertRtolEqual(cpu_output, npu_output)

    def test_solve_float16_1(self, device):
        def cpu_op_exec_float16_1(input1, input2):
            input1 = input1.to(torch.float32)
            input2 = input2.to(torch.float32)
            X, LU = torch.solve(input2, input1)
            X = X.numpy()
            X = X.astype(np.float16)
            return X
        npu_input1 = self.generate_data(0, 100, (5, 5), np.float16)
        npu_input2 = self.generate_data(0, 100, (5, 5), np.float16)
        cpu_output = cpu_op_exec_float16_1(npu_input1, npu_input2)
        # npu_output = self.npu_op_exec(npu_input1, npu_input2)
        #self.assertRtolEqual(cpu_output, npu_output)

    def test_solve_float32_1(self, device):
        npu_input1 = self.generate_data(0, 100, (2, 3, 2, 2), np.float32)
        npu_input2 = self.generate_data(0, 100, (2, 1, 2, 1), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        # npu_output = self.npu_op_exec(npu_input1, npu_input2)
        # self.assertRtolEqual(cpu_output, npu_output)

    def test_solve_float32_2(self, device):
        npu_input1 = self.generate_data(0, 100, (3, 3, 3), np.float32)
        npu_input2 = self.generate_data(0, 100, (3, 3, 2), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        # npu_output = self.npu_op_exec(npu_input1, npu_input2)
        # self.assertRtolEqual(cpu_output, npu_output)

  
instantiate_device_type_tests(TestSolve, globals(), except_for='cpu')
if __name__ == '__main__':
    run_tests()