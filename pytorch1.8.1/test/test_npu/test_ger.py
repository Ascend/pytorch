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
import copy
import torch.nn as nn
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestGer(TestCase):

    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype)

        # modify from numpy.ndarray to torch.tensor
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)

        return npu_input1, npu_input2

    def cpu_op_exec(self,input1, input2):
        output = torch.ger(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec(self,input1, input2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        output = torch.ger(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_add_float32(self, device):
        npu_input1, npu_input2 = self.generate_data(0, 100, (4), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test_add_float32(self, device):
        npu_input1, npu_input2 = self.generate_data(0, 100, (15), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)
        
    def test_add_float32(self, device):
        npu_input1, npu_input2 = self.generate_data(0, 100, (128), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)
    

    def test_add_float16(self, device):
        npu_input1, npu_input2 = self.generate_data(0, 100, (4), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1.to(torch.float32), npu_input2.to(torch.float32))
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output.astype(np.float16), npu_output.astype(np.float16))
        
    def test_add_float16(self, device):
        npu_input1, npu_input2 = self.generate_data(0, 100, (15), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1.to(torch.float32), npu_input2.to(torch.float32))
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output.astype(np.float16), npu_output.astype(np.float16))
        
    def test_add_float16(self, device):
        npu_input1, npu_input2 = self.generate_data(0, 100, (128), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1.to(torch.float32), npu_input2.to(torch.float32))
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output.astype(np.float16), npu_output.astype(np.float16))


instantiate_device_type_tests(TestGer, globals(), except_for="cpu")
if __name__ == "__main__":
    torch.npu.set_device("npu:2")
    run_tests()