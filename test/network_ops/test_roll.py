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


class TestRoll(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input_x = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input = torch.from_numpy(input_x)
        return npu_input

    def cpu_op_exec(self, input_x, shifts, dims):
        output = torch.roll(input_x, shifts, dims).numpy()
        return output

    def npu_op_exec(self, input_x, shifts, dims):
        input1 = input_x.to("npu")
        output = torch.roll(input1, shifts, dims)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_roll_3_4_5_float32(self):
        input_x1 = self.generate_data(-1, 1, (3, 4, 5), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, [2, 1], [0, 1])
        npu_output1 = self.npu_op_exec(input_x1, [2, 1], [0, 1])
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_roll_3_4_5_float16(self):
        input_x1 = self.generate_data(-1, 1, (3, 4, 5), np.float16)
        input_cpu = input_x1.float()
        cpu_output1 = self.cpu_op_exec(input_cpu, [2, 1], [0, 1]).astype(np.float16)
        npu_output1 = self.npu_op_exec(input_x1, [2, 1], [0, 1])
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_roll_30_40_50_int32(self):
        input_x1 = self.generate_data(-1, 1, (30, 40, 50), np.int32)
        cpu_output1 = self.cpu_op_exec(input_x1, [20], [])
        npu_output1 = self.npu_op_exec(input_x1, [20], [])
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_roll_20_30_40_50_uint8(self):
        input_x1 = self.generate_data(-1, 1, (20, 30, 40, 50), np.uint8)
        cpu_output1 = self.cpu_op_exec(input_x1, [-20, 30], [-1, 0])
        npu_output1 = self.npu_op_exec(input_x1, [-20, 30], [-1, 0])
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_roll_20_30_40_50_flaot32(self):
        input_x1 = self.generate_data(-1, 1, (20, 30, 40, 50), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1, [30], [3])
        npu_output1 = self.npu_op_exec(input_x1, [30], [3])
        self.assertRtolEqual(cpu_output1, npu_output1)


if __name__ == "__main__":
    run_tests()
