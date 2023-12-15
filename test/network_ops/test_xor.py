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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestXor(TestCase):

    def generate_bool_data(self, shape):
        input1 = np.random.uniform(0, 1, shape)
        input2 = np.random.uniform(0, 1, shape)
        input1 = input1.reshape(-1)
        input2 = input2.reshape(-1)
        len1 = len(input1)
        len2 = len(input2)

        for i in range(len1):
            if input1[i] < 0.5:
                input1[i] = 0
        for i in range(len2):
            if input2[i] < 0.5:
                input2[i] = 0
        input1 = input1.astype(np.bool_)
        input2 = input2.astype(np.bool_)
        input1 = input1.reshape(shape)
        input2 = input2.reshape(shape)
        # modify from numpy.ndarray to torch.tensor
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)

        return npu_input1, npu_input2

    def generate_single_bool_data(self, shape):
        input1 = np.random.uniform(0, 1, shape)
        input1 = input1.reshape(-1)
        len3 = len(input1)
        for i in range(len3):
            if input1[i] < 0.5:
                input1[i] = 0
        input1 = input1.astype(np.bool_)
        input1 = input1.reshape(shape)
        # modify from numpy.ndarray to torch.tensor
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

    def cpu_op_exec(self, input1, input2):
        output = input1 ^ input2
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        output = input1.__xor__(input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_scalar(self, input1, input2):
        input1 = input1.to("npu")
        output = input1.__xor__(input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_xor_tensor_int32(self, device="npu"):
        npu_input1 = self.generate_single_data(0, 100, (10, 10), np.int32)
        npu_input2 = self.generate_single_data(0, 100, (10, 10), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertEqual(cpu_output, npu_output)

    def test_xor_tensor_int16(self, device="npu"):
        npu_input1 = self.generate_single_data(0, 100, (10, 10), np.int16)
        npu_input2 = self.generate_single_data(0, 100, (10, 10), np.int16)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertEqual(cpu_output, npu_output)

    def test_xor_tensor_int8(self, device="npu"):
        npu_input1 = self.generate_single_data(0, 100, (10, 10), np.int8)
        npu_input2 = self.generate_single_data(0, 100, (10, 10), np.int8)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertEqual(cpu_output, npu_output)

    def test_xor_scalar_int32(self, device="npu"):
        npu_input = self.generate_single_data(0, 100, (1, 10), np.int32)
        npu_input_scalr = np.random.randint(0, 100)
        cpu_output = self.cpu_op_exec(npu_input, npu_input_scalr)
        npu_output = self.npu_op_exec_scalar(npu_input, npu_input_scalr)
        self.assertEqual(cpu_output, npu_output)

    def test_xor_scalar_int16(self, device="npu"):
        npu_input = self.generate_single_data(0, 100, (10, 20), np.int16)
        npu_input_scalr = np.random.randint(0, 100)
        cpu_output = self.cpu_op_exec(npu_input, npu_input_scalr)
        npu_output = self.npu_op_exec_scalar(npu_input, npu_input_scalr)
        self.assertEqual(cpu_output, npu_output)

    def test_xor_scalar_int8(self, device="npu"):
        npu_input = self.generate_single_data(0, 100, (20, 10), np.int8)
        npu_input_scalr = np.random.randint(0, 100)
        cpu_output = self.cpu_op_exec(npu_input, npu_input_scalr)
        npu_output = self.npu_op_exec_scalar(npu_input, npu_input_scalr)
        self.assertEqual(cpu_output, npu_output)

    def test_xor_tensor_uint8(self, device="npu"):
        npu_input1 = self.generate_single_data(0, 100, (10, 10), np.uint8)
        npu_input2 = self.generate_single_data(0, 100, (10, 10), np.uint8)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertEqual(cpu_output, npu_output)

    def test_xor_scalar_uint8(self, device="npu"):
        npu_input = self.generate_single_data(0, 100, (5, 10), np.uint8)
        npu_input_scalr = np.random.randint(0, 100)
        cpu_output = self.cpu_op_exec(npu_input, npu_input_scalr)
        npu_output = self.npu_op_exec_scalar(npu_input, npu_input_scalr)
        self.assertEqual(cpu_output, npu_output)

    def test_xor_scalar_bool1(self, device="npu"):
        npu_input = self.generate_single_bool_data((10, 10))
        npu_input_scalr = True
        cpu_output = self.cpu_op_exec(npu_input, npu_input_scalr)
        npu_output = self.npu_op_exec_scalar(npu_input, npu_input_scalr)
        self.assertEqual(cpu_output, npu_output)

    def test_xor_scalar_bool2(self, device="npu"):
        npu_input = self.generate_single_bool_data((10, 10))
        npu_input_scalr = False
        cpu_output = self.cpu_op_exec(npu_input, npu_input_scalr)
        npu_output = self.npu_op_exec_scalar(npu_input, npu_input_scalr)
        self.assertEqual(cpu_output, npu_output)

    def test_xor_tensor_bool(self, device="npu"):
        npu_input1, npu_input2 = self.generate_bool_data((10, 10))
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
