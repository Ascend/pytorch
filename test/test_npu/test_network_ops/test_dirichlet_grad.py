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
from torch.autograd import Variable
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestDirichletGrad(TestCase):
    def generate_data(self, min, max, shape, dtype):
        input1 = np.random.uniform(min, max, shape).astype(dtype)
        input2 = np.random.uniform(min, max, shape).astype(dtype)
        input3 = np.random.uniform(min, max, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
        npu_input3 = torch.from_numpy(input3)
        return npu_input1, npu_input2, npu_input3
        
    def cpu_op_exec(self, input1, input2, input3):
        output = torch._dirichlet_grad(input1, input2, input3)
        return output

    def npu_op_exec(self, input1, input2, input3):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        input3 = input3.to("npu")
        output = torch._dirichlet_grad(input1, input2, input3)
        output = output.to("cpu")
        return output

    def test_symeig_float(self, device):
        npu_input1, npu_input2, npu_input3 = self.generate_data(0, 100, (5, 5), np.float32)
        cpu_output1 = self.cpu_op_exec(npu_input1, npu_input2, npu_input3)
        # npu_output1 = self.npu_op_exec(npu_input1, npu_input2, npu_input3)
        # self.assertRtolEqual(cpu_output1, npu_output1)
        npu_input1, npu_input2, npu_input3 = self.generate_data(0, 100, (10, 5, 5), np.float64)
        cpu_output2 = self.cpu_op_exec(npu_input1, npu_input2, npu_input3)
        # npu_output2 = self.npu_op_exec(npu_input1, npu_input2, npu_input3)
        # self.assertRtolEqual(cpu_output2, npu_output2)
        npu_input1, npu_input2, npu_input3 = self.generate_data(0, 100, (10, 3, 5, 5), np.float64)
        cpu_output3 = self.cpu_op_exec(npu_input1, npu_input2, npu_input3)
        # npu_output3 = self.npu_op_exec(npu_input1, npu_input2, npu_input3)
        # self.assertRtolEqual(cpu_output3, npu_output3)
        npu_input1, npu_input2, npu_input3 = self.generate_data(0, 100, (2, 10, 3, 5, 5), np.float64)
        cpu_output4 = self.cpu_op_exec(npu_input1, npu_input2, npu_input3)
        # npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_input3)
        # self.assertRtolEqual(cpu_output4, npu_output4)
  

instantiate_device_type_tests(TestDirichletGrad, globals(), except_for='cpu')
if __name__ == '__main__':
    run_tests()