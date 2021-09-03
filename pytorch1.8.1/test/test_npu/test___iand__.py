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


class Test__Iand__(TestCase):

    def generate_bool_data(self, shape):
        input1 = np.random.uniform(0, 1, shape).astype(np.float32)
        input1 = input1 < 0.5
        npu_input1 = torch.from_numpy(input1)

        return npu_input1
    
    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype)

        # modify from numpy.ndarray to torch.tensor
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
        
        return npu_input1, npu_input2
        

    def generate_single_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        
        return npu_input1

    def generate_scalar(self, min_d, max_d):
        scalar = np.random.uniform(min_d, max_d)
        return scalar

    def generate_int_scalar(self, min_d, max_d):
        scalar = np.random.randint(min_d, max_d)
        return scalar

    def cpu_op_exec(self, input1, input2):
        input1 = input1.to("cpu")
        input2 = input2.to("cpu")
        output = input1.__iand__(input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_scalar(self, input1, input2):
        input1 = input1.to("cpu")
        output = input1.__iand__(input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        output = input1.__iand__(input2)
        output = output.to("cpu")
        output = output.numpy()
        return output    
        
    def npu_op_exec_scalar(self, input1, input2):
        input1 = input1.to("npu")
        output = input1.__iand__(input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test___iand___bool(self, device):
        npu_input1, npu_input2 = self.generate_bool_data((3, 5)), self.generate_bool_data((3, 5))
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test___iand___int16(self, device):
        npu_input1, npu_input2= self.generate_data(0, 100, (4, 3), np.int16)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        cpu_output = cpu_output.astype(np.int32)
        npu_output = npu_output.astype(np.int32)
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test___iand___int32(self, device):
        npu_input1, npu_input2= self.generate_data(0, 100, (4, 3), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        cpu_output = cpu_output.astype(np.int32)
        npu_output = npu_output.astype(np.int32)
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test___iand___scalar_bool(self, device):
        npu_input1 = self.generate_bool_data((3, 5))
        cpu_output = self.cpu_op_exec_scalar(npu_input1, True)
        npu_output = self.npu_op_exec_scalar(npu_input1, True)
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test___iand___scalar_int16(self, device):
        npu_input1 = self.generate_single_data(0, 100, (4, 3), np.int16)
        cpu_output = self.cpu_op_exec_scalar(npu_input1, 1)
        npu_output = self.npu_op_exec_scalar(npu_input1, 1)
        cpu_output = cpu_output.astype(np.int32)
        npu_output = npu_output.astype(np.int32)
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test___iand___scalar_int32(self, device):
        npu_input1 = self.generate_single_data(0, 100, (4, 3), np.int32)
        cpu_output = self.cpu_op_exec_scalar(npu_input1, 1)
        npu_output = self.npu_op_exec_scalar(npu_input1, 1)
        cpu_output = cpu_output.astype(np.int32)
        npu_output = npu_output.astype(np.int32)
        self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(Test__Iand__, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:1")
    run_tests()