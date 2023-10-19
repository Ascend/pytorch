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


class TestSelu(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input_x = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input = torch.from_numpy(input_x)
        return npu_input

    def cpu_op_exec(self, input_x):
        output = torch.selu(input_x)
        output = output.numpy()
        return output.astype(np.float32)

    def npu_op_exec(self, input_x):
        input1 = input_x.to("npu")
        output = torch.selu(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_inplace(self, input_x):
        input_x = input_x.to("npu")
        torch.selu_(input_x)
        output = input_x.to("cpu")
        output = output.numpy()
        return output

    def test_selu_3_3(self):
        input_x1 = self.generate_data(-1, 1, (3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1)
        npu_output1 = self.npu_op_exec(input_x1)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_selu_3_3_3_3(self):
        input_x1 = self.generate_data(-1, 1, (3, 3, 3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1)
        npu_output1 = self.npu_op_exec(input_x1)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_selu_3_3_float16(self):
        input_x1 = self.generate_data(-1, 1, (3, 3), np.float16)
        input_x1_cpu = input_x1.float()
        cpu_output1 = self.cpu_op_exec(input_x1_cpu).astype(np.float16)
        npu_output1 = self.npu_op_exec(input_x1)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_selu_3_3_3_3_float16(self):
        input_x1 = self.generate_data(-1, 1, (3, 3, 3, 3), np.float16)
        input_x1_cpu = input_x1.float()
        cpu_output1 = self.cpu_op_exec(input_x1_cpu).astype(np.float16)
        npu_output1 = self.npu_op_exec(input_x1)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_selu_3_3_inplace(self):
        input_x1 = self.generate_data(-1, 1, (3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1)
        npu_output1 = self.npu_op_exec_inplace(input_x1)
        self.assertRtolEqual(cpu_output1, npu_output1)

    def test_selu_3_3_3_3_inplace(self):
        input_x1 = self.generate_data(-1, 1, (3, 3, 3, 3), np.float32)
        cpu_output1 = self.cpu_op_exec(input_x1)
        npu_output1 = self.npu_op_exec_inplace(input_x1)
        self.assertRtolEqual(cpu_output1, npu_output1)


if __name__ == "__main__":
    run_tests()
