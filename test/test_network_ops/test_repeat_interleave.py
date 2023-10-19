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
from torch_npu.testing.common_utils import create_common_tensor


class TestRepeatInterleave(TestCase):

    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        cpu_input1 = torch.from_numpy(input1)
        return cpu_input1

    def cpu_op_exec(self, input1, input2, input3):
        output = torch.repeat_interleave(input1, input2, dim=input3)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2, input3):
        output = torch.repeat_interleave(input1, input2, dim=input3)
        output = output.cpu()
        output = output.numpy()
        return output

    def cpu_op_exec_without_dim(self, input1, input2):
        output = torch.repeat_interleave(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec_without_dim(self, input1, input2):
        output = torch.repeat_interleave(input1, input2)
        output = output.cpu()
        output = output.numpy()
        return output

    def cpu_op_exec_tensor(self, input1, input2, input3):
        output = torch.repeat_interleave(input1, input2, dim=input3)
        output = output.numpy()
        return output

    def npu_op_exec_tensor(self, input1, input2, input3):
        output = torch.repeat_interleave(input1, input2, dim=input3)
        output = output.cpu()
        output = output.numpy()
        return output

    def test_shape_format_tensor_int64(self):
        format_list = [2]
        shape_list = [[2, 7, 3]]
        repeats_list = [[4, 2, 3]]
        dim_list = [2]
        shape_format = [
            [[np.int64, i, j], p, v]
            for i in format_list
            for j in shape_list
            for p in repeats_list
            for v in dim_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_repeat = torch.tensor(item[1])
            npu_repeat = cpu_repeat.npu()
            cpu_output = self.cpu_op_exec_tensor(cpu_input1, cpu_repeat, item[2])
            npu_output = self.npu_op_exec_tensor(npu_input1, npu_repeat, item[2])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_shape_format_tensor_int32(self):
        format_list = [2]
        shape_list = [[2, 7, 3]]
        repeats_list = [4, 2, 3]
        dim_list = [2]
        shape_format = [
            [[np.int32, i, j], p, v]
            for i in format_list
            for j in shape_list
            for p in repeats_list
            for v in dim_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1], item[2])
            npu_output = self.npu_op_exec(npu_input1, item[1], item[2])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_repeat_interleave_float16(self):
        cpu_input1 = self.generate_data(0, 100, (3, 3, 3), np.float16)
        input2 = np.random.randint(1, 100)
        input3 = np.random.randint(0, 2)
        cpu_output = self.cpu_op_exec(cpu_input1, input2, input3)
        npu_output = self.npu_op_exec(cpu_input1.npu(), input2, input3)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_repeat_interleave_float32(self):
        cpu_input1 = self.generate_data(0, 100, (3, 3, 3), np.float32)
        input2 = np.random.randint(1, 100)
        input3 = np.random.randint(0, 2)
        cpu_output = self.cpu_op_exec(cpu_input1, input2, input3)
        npu_output = self.npu_op_exec(cpu_input1.npu(), input2, input3)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_repeat_interleave_int32(self):
        cpu_input1 = self.generate_data(0, 100, (3, 3, 3), np.int32)
        input2 = np.random.randint(1, 100)
        input3 = np.random.randint(0, 2)
        cpu_output = self.cpu_op_exec(cpu_input1, input2, input3)
        npu_output = self.npu_op_exec(cpu_input1.npu(), input2, input3)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_repeat_interleave_int32_without_dim(self):
        cpu_input1 = self.generate_data(0, 100, (3, 3, 3), np.int32)
        input2 = np.random.randint(1, 100)
        cpu_output = self.cpu_op_exec_without_dim(cpu_input1, input2)
        npu_output = self.npu_op_exec_without_dim(cpu_input1.npu(), input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_repeat_interleave_repeats_contains_one_ele(self):
        format_list = [2]
        shape_list = [[2, 7, 3]]
        repeats_list = [4]
        dim_list = [2]
        shape_format = [
            [[np.int64, i, j], p, v]
            for i in format_list
            for j in shape_list
            for p in repeats_list
            for v in dim_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_repeat = torch.tensor(item[1])
            npu_repeat = cpu_repeat.npu()
            cpu_output = self.cpu_op_exec_tensor(cpu_input1, cpu_repeat, item[2])
            npu_output = self.npu_op_exec_tensor(npu_input1, npu_repeat, item[2])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
