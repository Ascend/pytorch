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


class TestFill(TestCase):

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
        output = torch.fill_(input1, input2).numpy()
        return output

    def npu_op_exec(self, input1, input2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        output = torch.fill_(input1, input2)
        output = output.to("cpu").numpy()
        return output

    def npu_op_exec_scalar(self, input1, input2):
        input1 = input1.to("npu")
        output = torch.fill_(input1, input2)
        output = output.to("cpu").numpy()
        return output


    def test_fill_scalar_int32(self, device):
        npu_input1, _ = self.generate_data(0, 100, (2, 3), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, 1)
        npu_output = self.npu_op_exec_scalar(npu_input1, 1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_fill_scalar_float16(self, device):
        npu_input1, _ = self.generate_data(0, 100, (2, 3), np.float16)
        cpu_output = self.cpu_op_exec(npu_input1, 1)
        npu_output = self.npu_op_exec_scalar(npu_input1, 1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_fill_scalar_float32(self, device):
        npu_input1, _ = self.generate_data(0, 100, (2, 3), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, 1)
        npu_output = self.npu_op_exec_scalar(npu_input1, 1)
        self.assertRtolEqual(cpu_output, npu_output)


    def test_fill_common_shape_format(self, device):
        shape_format = [
            [np.float32, -1, (4, 3)],
            [np.int32, -1, (2, 3)],
            [np.int32, -1, (4, 3, 1)],
            [np.float16,-1,(65535, 1)],
            [np.float16, -1, (1, 8192)],
            [np.float16, -1, (1, 16384)],
            [np.float16, -1, (1, 32768)],
            [np.float16, -1, ( 1, 131072)],
            [np.float16, -1, (1, 196608)],
            [np.float16, -1, (1, 262144)],
            [np.float16, -1, (1, 393216)],
            [np.float16, -1, (1, 524288)],
            [np.float16, -1, (1, 655360)],
            [np.float16, -1, (1, 786432)],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, 1)
            npu_output = self.npu_op_exec_scalar(npu_input1, 1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_fill_float32_data_range(self, device):
        data_range = [
            [-1.1754943508e-38, -1.1754943508e-38],
            [-3402823500.0, 3402823500.0],
            [-0.000030517578125, 0.000030517578125],
            [3402823500, 3402800000],
            [-9.313225746154785e-10, 9.313225746154785e-10],
            [-3402823500.0, -3402823500.0],
            [-3402823500.0, 3402823500.0],
            [-9.313225746154785e-10, 9.313225746154785e-10],
            [-3402823500.0,-3402823500.0],
            [-0.000000000000000000000000000000000000011754943508, 0.000000000000000000000000000000000000011754943508],
            [0.000000000000000000000000000000000000011754943508, 0.000000000000000000000000000000000000011754943508],
            [-0.000000000000000000000000000000000000011754943508, -0.000000000000000000000000000000000000011754943508],
            [-0.000000000000000000000000000000000000011754943508, 0.000000000000000000000000000000000000011754943508]
        ]
        for item in data_range:
            cpu_input1, npu_input1 = create_common_tensor([np.float32, - 1, (1, 31, 149, 2)], item[0], item[1])
            cpu_output = self.cpu_op_exec(cpu_input1, 1)
            npu_output = self.npu_op_exec_scalar(npu_input1, 1)
            self.assertRtolEqual(cpu_output, npu_output)
            print("float32 run")

instantiate_device_type_tests(TestFill, globals(), except_for='cpu')
if __name__ == '__main__':
    torch.npu.set_device("npu:7")
    run_tests()
