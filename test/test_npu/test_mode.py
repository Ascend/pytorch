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

import copy
import numpy as np
import torch
import torch.nn as nn
from common_device_type import dtypes, instantiate_device_type_tests
from common_utils import TestCase, run_tests
from util_test import create_common_tensor

#pylint: disable=import-error
#pylint: disable=too-many-lines
#pylint: disable=too-many-arguments
#pylint: disable=unused-variable
#pylint: disable=unused-argument

class TestMode(TestCase):

    def generate_data_1(self, dtype):
        input = np.array([[10, 11, 12, 11, 10, 10, 10, 11],
                          [11, 11, 11, 10, 11, 10, 10, 11],
                          [12, 10, 10, 12, 10, 11, 10, 13],
                          [12, 10, 11, 12, 11, 11, 10, 13],
                          [14, 11, 11, 12, 10, 11, 10, 13]]).astype(np.float32)

        # modify from numpy.ndarray to torch.tensor
        npu_input = torch.from_numpy(input)
        return npu_input

    def generate_data_2(self, dtype):
        input = np.array([[10, 11, 12, 11, 10],
                          [11, 10, 11, 10, 11],
                          [12, 10, 10, 12, 10],
                          [12, 10, 11, 13, 11],
                          [14, 11, 11, 12, 10]]).astype(np.float32)

        # modify from numpy.ndarray to torch.tensor
        npu_input = torch.from_numpy(input)
        return npu_input
    
    
    def generate_data_3(self, dtype):
         input = np.zeros((36,25)).astype(np.float32)
         input[:,2]=1
         input[:,12]=2
         # modify from numpy.ndarray to torch.tensor
         npu_input = torch.from_numpy(input)
         return npu_input
    
    def generate_data_4(self, dtype):
        input = np.zeros((12,12)).astype(np.float32)
        # modify from numpy.ndarray to torch.tensor
        input[:,2]=1
        input[:,10]=2
        npu_input = torch.from_numpy(input)
        return npu_input
        
    def cpu_op_exec_0(self, input):
        output1, output2 = torch.mode(input, 0, keepdim=False)
        output1 = output1.numpy()
        output2 = output2.numpy()

        return output1, output2

    def npu_op_exec_0(self, input):
        input = input.to("npu")
        output1, output2 = torch.mode(input, 0, keepdim=False)
        output1 = output1.to("cpu")
        output2 = output2.to("cpu")
        output1 = output1.numpy()
        output2 = output2.numpy()

        return output1, output2

    def cpu_op_exec_1(self, input):
        output1, output2 = torch.mode(input, 1, keepdim=False)
        output1 = output1.numpy()
        output2 = output2.numpy()

        return output1, output2
        
    def npu_op_exec_1(self, input):
        input = input.to("npu")
        output1, output2 = torch.mode(input, 1, keepdim=False)
        output1 = output1.to("cpu")
        output2 = output2.to("cpu")
        output1 = output1.numpy()
        output2 = output2.numpy()

        return output1, output2
        
    def _cpu_op_exec_0(self, input):
        output1, output2 = torch._mode(input, 0, keepdim=False)
        output1 = output1.numpy()
        output2 = output2.numpy()

        return output1, output2
    
    def _npu_op_exec_0(self, input):
        input = input.to("npu")
        output1, output2 = torch._mode(input, 0, keepdim=False)
        output1 = output1.to("cpu")
        output2 = output2.to("cpu")
        output1 = output1.numpy()
        output2 = output2.numpy()

        return output1, output2
        
    def _cpu_op_exec_1(self, input):
        output1, output2 = torch._mode(input, 1, keepdim=False)
        output1 = output1.numpy()
        output2 = output2.numpy()

        return output1, output2
    
    def _npu_op_exec_1(self, input):
        input = input.to("npu")
        output1, output2 = torch._mode(input, 1, keepdim=False)
        output1 = output1.to("cpu")
        output2 = output2.to("cpu")
        output1 = output1.numpy()
        output2 = output2.numpy()

        return output1, output2

    def test_add_float32_0(self, device):
        npu_input = self.generate_data_1(np.float32)
        cpu_output1, cpu_output2 = self.cpu_op_exec_0(npu_input)
        npu_output1, npu_output2 = self.npu_op_exec_0(npu_input)
        self.assertRtolEqual(cpu_output1, npu_output1)
        self.assertRtolEqual(cpu_output2, npu_output2)

    def test_add_float32_1(self, device):
        npu_input = self.generate_data_2(np.float32)
        cpu_output1, cpu_output2 = self.cpu_op_exec_1(npu_input)
        npu_output1, npu_output2 = self.npu_op_exec_1(npu_input)
        self.assertRtolEqual(cpu_output1, npu_output1)
        self.assertRtolEqual(cpu_output2, npu_output2)

    def test_add_float32_2(self, device):
        npu_input = self.generate_data_3(np.float32)
        cpu_output1, cpu_output2 = self.cpu_op_exec_0(npu_input)
        npu_output1, npu_output2 = self.npu_op_exec_0(npu_input)
        self.assertRtolEqual(cpu_output1, npu_output1)
        self.assertRtolEqual(cpu_output2, npu_output2)

    def test_add_float32_3(self, device):
        npu_input = self.generate_data_4(np.float32)
        cpu_output1, cpu_output2 = self.cpu_op_exec_0(npu_input)
        npu_output1, npu_output2 = self.npu_op_exec_0(npu_input)
        self.assertRtolEqual(cpu_output1, npu_output1)
        self.assertRtolEqual(cpu_output2, npu_output2)
        
    def test_add_float32_4(self, device):
        npu_input = self.generate_data_2(np.float32)
        cpu_output1, cpu_output2 = self._cpu_op_exec_0(npu_input)
        npu_output1, npu_output2 = self._npu_op_exec_0(npu_input)
        self.assertRtolEqual(cpu_output1, npu_output1)
        self.assertRtolEqual(cpu_output2, npu_output2)
        
    def test_add_float32_5(self, device):
        npu_input = self.generate_data_1(np.float32)
        cpu_output1, cpu_output2 = self._cpu_op_exec_1(npu_input)
        npu_output1, npu_output2 = self._npu_op_exec_1(npu_input)
        self.assertRtolEqual(cpu_output1, npu_output1)
        self.assertRtolEqual(cpu_output2, npu_output2)
    
    def test_add_float32_6(self, device):
        npu_input = self.generate_data_3(np.float32)
        cpu_output1, cpu_output2 = self._cpu_op_exec_0(npu_input)
        npu_output1, npu_output2 = self._npu_op_exec_0(npu_input)
        self.assertRtolEqual(cpu_output1, npu_output1)
        self.assertRtolEqual(cpu_output2, npu_output2)
        
    def test_add_float32_7(self, device):
        npu_input = self.generate_data_4(np.float32)
        cpu_output1, cpu_output2 = self._cpu_op_exec_1(npu_input)
        npu_output1, npu_output2 = self._npu_op_exec_1(npu_input)
        self.assertRtolEqual(cpu_output1, npu_output1)
        self.assertRtolEqual(cpu_output2, npu_output2)
        
instantiate_device_type_tests(TestMode, globals(), except_for="cpu")
if __name__ == "__main__":
    torch.npu.set_device("npu:2")
    run_tests()