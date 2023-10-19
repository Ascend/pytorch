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
import copy
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestFloorDivide(TestCase):

    def generate_data(self, min1, max1, shape, dtype):
        input1 = np.random.uniform(min1, max1, shape).astype(dtype)
        input2 = np.random.uniform(min1, max1, shape).astype(dtype)

        # modify from numpy.ndarray to torch.tensor
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)

        return npu_input1, npu_input2

    def generate_three_data(self, min1, max1, shape, dtype):
        input1 = np.random.uniform(min1, max1, shape).astype(dtype)
        input2 = np.random.uniform(min1, max1, shape).astype(dtype)
        input3 = np.random.uniform(min1, max1, shape).astype(dtype)

        # modify from numpy.ndarray to torch.tensor
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
        npu_input3 = torch.from_numpy(input3)

        return npu_input1, npu_input2, npu_input3

    def cpu_op_exec(self, input1, input2):
        output = torch.floor_divide(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        output = torch.floor_divide(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_scalar(self, input1, input2):
        input1 = input1.to("npu")
        output = torch.floor_divide(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2, input3):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        output = input3.to("npu")
        torch.floor_divide(input1, input2, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_floor_divide_float32(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(1, 100, (1, 2), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_floor_divide_float32_out(self, device="npu"):
        npu_input1, npu_input2, npu_input3 = self.generate_three_data(1, 100, (1, 2), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_floor_divide_int32(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(1, 100, (1, 2), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_floor_divide_int8(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(1, 100, (1, 2), np.int8)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_floor_divide_uint8(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(1, 100, (1, 3), np.uint8)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_floor_divide_scalar_float32(self, device="npu"):
        npu_input1, _ = self.generate_data(1, 100, (1, 3), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, 1)
        npu_output = self.npu_op_exec_scalar(npu_input1, 1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_floor_divide_scalar_bool(self, device="npu"):
        npu_input1, _ = self.generate_data(1, 10, (2, 5), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1 > 5, 1.0)
        npu_output = self.npu_op_exec_scalar(npu_input1 > 5, 1.0)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_floor_divide_scalar_int32(self, device="npu"):
        npu_input1, _ = self.generate_data(1, 10, (4, 6), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, 1.5)
        npu_output = self.npu_op_exec_scalar(npu_input1, 1.5)
        self.assertRtolEqual(cpu_output, npu_output)

    def npu_uncontiguous_op_exec_scalar(self, input1, input2):
        input1 = input1.to("npu")
        input1 = input1.as_strided([2, 2], [1, 2], 1)
        output = torch.floor_divide(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_uncontiguous_op_exec_scalar(self, input1, input2):
        input1 = input1.as_strided([2, 2], [1, 2], 1)
        output = torch.floor_divide(input1, input2)
        output = output.numpy()
        return output

    def test_floor_divide_uncontiguous_float32_scalar(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(1, 100, (4, 3), np.float32)
        cpu_input1 = copy.deepcopy(npu_input1)
        cpu_output = self.cpu_uncontiguous_op_exec_scalar(cpu_input1, 2)
        npu_output = self.npu_uncontiguous_op_exec_scalar(npu_input1, 2)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
