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


class TestRepeatInterleave(TestCase):

    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        return npu_input1

    def cpu_op_exec(self, input1, input2, input3):
        output = torch.repeat_interleave(input1, input2, dim=input3)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2, input3):
        output = torch.repeat_interleave(input1, input2, dim=input3)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_without_dim(self, input1, input2):
        output = torch.repeat_interleave(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec_without_dim(self, input1, input2):
        output = torch.repeat_interleave(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_repeat_interleave_float16(self, device="npu"):
        npu_input1 = self.generate_data(0, 100, (3, 3, 3), np.float16)
        npu_input2 = np.random.randint(1, 100)
        npu_input3 = np.random.randint(0, 2)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, npu_input3)
        npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_input3)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_repeat_interleave_float32(self, device="npu"):
        npu_input1 = self.generate_data(0, 100, (3, 3, 3), np.float32)
        npu_input2 = np.random.randint(1, 100)
        npu_input3 = np.random.randint(0, 2)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, npu_input3)
        npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_input3)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_repeat_interleave_int32(self, device="npu"):
        npu_input1 = self.generate_data(0, 100, (3, 3, 3), np.int32)
        npu_input2 = np.random.randint(1, 100)
        npu_input3 = np.random.randint(0, 2)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, npu_input3)
        npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_input3)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_repeat_interleave_int32_without_dim(self, device="npu"):
        npu_input1 = self.generate_data(0, 100, (3, 3, 3), np.int32)
        npu_input2 = np.random.randint(1, 100)
        cpu_output = self.cpu_op_exec_without_dim(npu_input1, npu_input2)
        npu_output = self.npu_op_exec_without_dim(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()