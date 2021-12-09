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
from graph_utils import graph_mode


class TestSub(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype)

        # modify from numpy.ndarray to torch.tensor
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
        return npu_input1, npu_input2

    def cpu_op_exec(self, input1, input2):
        output = torch.sub(input1, input2)
        output = output.numpy()
        return output

    def cpu_op_exec_fp16(self, input1, input2):
        input1 = input1.to(torch.float32)
        output = torch.sub(input1, input2)
        output = output.numpy()
        output = output.astype(np.float16)
        return output

    def npu_op_exec(self, input1, input2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        output = torch.sub(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    @graph_mode
    def test_sub_int32_1(self, device):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2, 3), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    @graph_mode
    def test_sub_int32_2(self, device):
        npu_input1, npu_input2 = self.generate_data(2147483647, 2147483648, (2, 3), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    @graph_mode
    def test_sub_int32_3(self, device):
        npu_input1, npu_input2 = self.generate_data(-100, 0, (2, 3), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    @graph_mode
    def test_sub_int32_4(self, device):
        npu_input1, npu_input2 = self.generate_data(0, 100, (500, 500), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    @graph_mode
    def test_sub_float16_1(self, device):
        npu_input1, npu_input2 = self.generate_data(-2, 2, ((65535, 1, 1, 1)), np.float16)
        cpu_output = self.cpu_op_exec_fp16(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    @graph_mode
    def test_sub_float16_2(self, device):
        npu_input1, npu_input2 = self.generate_data(-2, 2, ((1, 1, 1, 8192)), np.float16)
        cpu_output = self.cpu_op_exec_fp16(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    @graph_mode
    def test_sub_float16_3(self, device):
        npu_input1, npu_input2 = self.generate_data(-2, 2, ((1, 1, 1, 65535)), np.float16)
        cpu_output = self.cpu_op_exec_fp16(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    @graph_mode
    def test_sub_float16_4(self, device):
        npu_input1, npu_input2 = self.generate_data(-2, 2, ((1, 1, 1, 524288)), np.float16)
        cpu_output = self.cpu_op_exec_fp16(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    @graph_mode
    def test_sub_float16_5(self, device):
        npu_input1, npu_input2 = self.generate_data(-2, 2, ((1, 1, 1, 786432)), np.float16)
        cpu_output = self.cpu_op_exec_fp16(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    @graph_mode
    def test_sub_float16_6(self, device):
        npu_input1, npu_input2 = self.generate_data(-5, 5, ((1, 1, 1, 786432)), np.float16)
        cpu_output = self.cpu_op_exec_fp16(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    @graph_mode
    def test_sub_float32_1(self, device):
        npu_input1, npu_input2 = self.generate_data(-1.1754943508e-38, -1.1754943508e-38, ((1, 31, 149, 2)), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    @graph_mode
    def test_sub_float32_2(self, device):
        npu_input1, npu_input2 = self.generate_data(-0.000030517578125, 0.000030517578125, ((2, 32, 149, 31)), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    @graph_mode
    def test_sub_float32_3(self, device):
        npu_input1, npu_input2 = self.generate_data(-9.313225746154785e-10, 9.313225746154785e-10, ((184965, 1)), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    @graph_mode
    def test_sub_float32_4(self, device):
        npu_input1, npu_input2 = self.generate_data(-3, 3, ((1, 31, 149, 2)), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    @graph_mode
    def test_sub_float32_5(self, device):
        npu_input1, npu_input2 = self.generate_data(-9.313225746154785e-10, 9.313225746154785e-10, ((1, 31, 149, 2)), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    @graph_mode
    def test_sub_float32_6(self, device):
        npu_input1, npu_input2 = self.generate_data(-0.000000000000000000000000000000000000011754943508,
                                          0.000000000000000000000000000000000000011754943508, ((2, 31, 149, 2)),
                                          np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    @graph_mode
    def test_sub_float32_7(self, device):
        npu_input1, npu_input2 = self.generate_data(0.000000000000000000000000000000000000011754943508,
                                          0.000000000000000000000000000000000000011754943508, ((4, 31, 149, 2)),
                                          np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    @graph_mode
    def test_sub_float32_8(self, device):
        npu_input1, npu_input2 = self.generate_data(-0.000000000000000000000000000000000000011754943508,
                                          -0.000000000000000000000000000000000000011754943508, ((2048, 31, 1, 2)),
                                          np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    @graph_mode
    def test_sub_float32_9(self, device):
        npu_input1, npu_input2 = self.generate_data(-0.000000000000000000000000000000000000011754943508,
                                          0.000000000000000000000000000000000000011754943508, ((8, 7, 149)), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    @graph_mode
    def test_sub_mix_dtype_1(self,device):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2, 3), np.int32)
        npu_input3, npu_input4 = self.generate_data(0, 100, (2, 3), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input3)
        npu_output = self.npu_op_exec(npu_input1, npu_input3)
        self.assertRtolEqual(cpu_output, npu_output)
        
    @graph_mode
    def test_sub_mix_dtype_2(self,device):
        npu_input1, npu_input2 = self.generate_data(0, 100, (2, 3), np.float32)
        npu_input3 = torch.tensor(3).int()
        cpu_output = self.cpu_op_exec(npu_input1, npu_input3)
        npu_output = self.npu_op_exec(npu_input1, npu_input3)
        self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestSub, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
