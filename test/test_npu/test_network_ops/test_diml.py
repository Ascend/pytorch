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
from torch.autograd import Variable
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestDiml(TestCase):
    def generate_data(self, min, max, shape, dtype):
        input1 = np.random.uniform(min, max, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        return npu_input1
        
    def cpu_op_exec(self, input1):
        input1[0][0] = 5
        input1_sparse = input1.to_sparse()
        outut = input1_sparse.indices().size(0)
        return outut 

    def npu_op_exec(self, input1):
        input1 = input1.to("npu")
        input1_sparse = input1.to_sparse()
        outut = input1_sparse.indices().size(0)
        outut = outut.to("cpu")
        return outut 

    def test_diml_float32_1(self, device):
        npu_input1 = self.generate_data(0, 100, (5, 5), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1)
        # npu_output = self.npu_op_exec(npu_input1)
        # self.assertRtolEqual(cpu_output, npu_output)
        
    def test_diml_float64_1(self, device):
        npu_input1 = self.generate_data(0, 100, (10, 5, 5), np.float64)
        cpu_output = self.cpu_op_exec(npu_input1)
        # npu_output = self.npu_op_exec(npu_input1)
        # self.assertRtolEqual(cpu_output, npu_output)

    def test_diml_float64_2(self, device):
        npu_input1 = self.generate_data(0, 100, (10, 3, 5, 5), np.float64)
        cpu_output = self.cpu_op_exec(npu_input1)
        # npu_output = self.npu_op_exec(npu_input1)
        # self.assertRtolEqual(cpu_output, npu_output)
         
    def test_diml_float64_3(self, device):
        npu_input1 = self.generate_data(0, 100, (2, 10, 3, 5, 5), np.float64)
        cpu_output = self.cpu_op_exec(npu_input1)
        # npu_output = self.npu_op_exec(npu_input1)
        # self.assertRtolEqual(cpu_output, npu_output)
  

instantiate_device_type_tests(TestDiml, globals(), except_for='cpu')
if __name__ == '__main__':
    run_tests()