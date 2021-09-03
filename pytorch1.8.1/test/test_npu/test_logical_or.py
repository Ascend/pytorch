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


class TestLogicalOr(TestCase):

    def generate_single_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)

        return npu_input1

    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype)

        #modify from numpy.ndarray to torch.tensor
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)

        return npu_input1, npu_input2
    
    def generate_three_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input3 = np.random.uniform(min_d, max_d, shape).astype(dtype)

        #modify from numpy.ndarray to torch.tensor
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
        npu_input3 = torch.from_numpy(input3)
        
        return npu_input1, npu_input2, npu_input3

    def generate_bool_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.randint(min_d, max_d, shape).astype(dtype)
        input2 = np.random.randint(min_d, max_d, shape).astype(dtype)

        #modify from numpy.ndarray to torch.tensor
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)

        return npu_input1, npu_input2

    def cpu_op_exec(self, input1, input2):
        output = torch.logical_or(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        output = torch.logical_or(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_out(self, input1, input2, input3):
        torch.logical_or(input1, input2, out=input3)
        output = input3.numpy()
        return output

    def npu_op_exec_out(self, input1, input2, input3):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        output = input3.to("npu")
        torch.logical_or(input1, input2, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output
    
    def cpu_op_exec_(self, input1, input2):
        output = torch.Tensor.logical_or_(input1, input2)
        output = output.numpy()
        return output
    
    def npu_op_exec_(self, input1, input2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        output = torch.Tensor.logical_or_(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output
    
    def test_logical_or_bool(self, device):
        npu_input1, npu_input2 = self.generate_bool_data(0, 2, (10, 64), np.bool)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_logical_or_uint8(self, device):
        npu_input1, npu_input2 = self.generate_data(0, 1000, (5, 12), np.uint8)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_logical_or_int8(self, device):
        npu_input1, npu_input2 = self.generate_data(-1000, 1000, (10, 12), np.int8)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test_logical_or_int32(self, device):
        npu_input1, npu_input2 = self.generate_data(-1000, 1000, (11, 64), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_logical_or_float16(self, device):
        npu_input1, npu_input2 = self.generate_data(0, 2, (15, 64), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_logical_or_float32(self, device):
        npu_input1, npu_input2 = self.generate_data(0, 2, (10, 64), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_logical_or_float32_broadcast(self, device):
        npu_input1 = self.generate_single_data(0, 2, (4,3,1), np.float32)
        npu_input2 = self.generate_single_data(0, 2, (4,1,5), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test_logical_or_int8_out(self, device):
        npu_input1, npu_input2, npu_input3 = self.generate_three_data(0, 2, (10,50), np.int8)
        cpu_output = self.cpu_op_exec_out(npu_input1, npu_input2, npu_input3)
        npu_output = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3)
        cpu_output = cpu_output.astype(np.int32)
        npu_output = npu_output.astype(np.int32)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_logical_or_float16_out(self, device):
        npu_input1, npu_input2, npu_input3 = self.generate_three_data(0, 2, (10,50), np.float16)
        cpu_output = self.cpu_op_exec_out(npu_input1, npu_input2, npu_input3)
        npu_output = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_logical_or_float32_out(self, device):
        npu_input1, npu_input2, npu_input3 = self.generate_three_data(0, 2, (10,50), np.float32)
        cpu_output = self.cpu_op_exec_out(npu_input1, npu_input2, npu_input3)
        npu_output = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3)
        self.assertRtolEqual(cpu_output, npu_output)
        
    def test_logical_or_float32_1D(self, device):
        npu_input1, npu_input2 = self.generate_data(0, 2, (10, ), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)
        
    def test_logical_or_float32_4D(self, device):
        npu_input1, npu_input2 = self.generate_data(0, 2, (10, 16, 12, 4), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_logical_or_float32_min(self, device):
        npu_input1 = torch.tensor([2**(-31), 1, 2**(-31), -2**(-31), 2**(-31)], dtype=torch.float32)
        npu_input2 = torch.tensor([0, 2**(-31), 2**(-31), 0, -2**(-31)], dtype=torch.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)
        
    def test_logical_or_inplace_bool(self, device):
        npu_input1, npu_input2 = self.generate_bool_data(0, 2, (10, 64), np.bool)
        cpu_output = self.cpu_op_exec_(npu_input1, npu_input2)
        npu_output = self.npu_op_exec_(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_logical_or_inplace_uint8(self, device):
        npu_input1, npu_input2 = self.generate_data(0, 1000, (5, 12), np.uint8)
        cpu_output = self.cpu_op_exec_(npu_input1, npu_input2)
        npu_output = self.npu_op_exec_(npu_input1, npu_input2)
        cpu_output = cpu_output.astype(np.int32)
        npu_output = npu_output.astype(np.int32)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_logical_or_inplace_int8(self, device):
        npu_input1, npu_input2 = self.generate_data(-1000, 1000, (10, 12), np.int8)
        cpu_output = self.cpu_op_exec_(npu_input1, npu_input2)
        npu_output = self.npu_op_exec_(npu_input1, npu_input2)
        cpu_output = cpu_output.astype(np.int32)
        npu_output = npu_output.astype(np.int32)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_logical_or_inplace_int32(self, device):
        npu_input1, npu_input2 = self.generate_data(-1000, 1000, (11, 64), np.int32)
        cpu_output = self.cpu_op_exec_(npu_input1, npu_input2)
        npu_output = self.npu_op_exec_(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_logical_or_inplace_float16(self, device):
        npu_input1, npu_input2 = self.generate_data(0, 2, (15, 64), np.float16)
        cpu_output = self.cpu_op_exec_(npu_input1, npu_input2)
        npu_output = self.npu_op_exec_(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_logical_or_inplace_float32(self, device):
        npu_input1, npu_input2 = self.generate_data(0, 2, (10, 64), np.float32)
        cpu_output = self.cpu_op_exec_(npu_input1, npu_input2)
        npu_output = self.npu_op_exec_(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)
    
instantiate_device_type_tests(TestLogicalOr, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:3")
    run_tests()