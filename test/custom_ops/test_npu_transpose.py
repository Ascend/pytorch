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

import unittest
import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestNpuTranspose(TestCase):

    def custom_op_exec(self, input1, perm):
        output = input1.permute(perm)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, perm):
        output = torch_npu.npu_transpose(input1, perm)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_complex_exec(self, input1, perm):
        return input1.permute(perm)

    def npu_complex_exec(self, input1, perm):
        output = torch_npu.npu_transpose(input1, perm)
        return output.to("cpu")

    def test_npu_transpose(self):
        shape_format = [
            [[np.float32, 0, (5, 3, 6, 4)], [1, 0, 2, 3]],
            [[np.float16, 0, (5, 3, 6, 4)], [0, 3, 2, 1]],
        ]

        for item in shape_format:
            _, npu_input1 = create_common_tensor(item[0], 0, 100)
            custom_output = self.custom_op_exec(npu_input1, item[1])
            npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(custom_output, npu_output)

    @unittest.skip("skip test_transpose_complex now")
    def test_transpose_complex(self):
        shape_format = [
            [[np.float32, 0, (5, 3)], [1, 0]],
            [[np.float16, 0, (5, 3)], [0, 1]],
            [[np.float32, 0, (5, 3)], [-1, 0]],
            [[np.float16, 0, (5, 3)], [0, -1]],
            [[np.float32, 0, (4000, 4000)], [-1, 0]],
            [[np.float16, 0, (4000, 4000)], [0, -1]],
            [[np.float32, 0, (5, 3, 6, 4)], [1, 0, 2, 3]],
            [[np.float16, 0, (5, 3, 6, 4)], [0, 3, 2, 1]],
            [[np.float32, 0, (5, 3, 6, 4)], [-3, 0, -2, -1]],
            [[np.float16, 0, (5, 3, 6, 4)], [0, -1, -2, -3]],
        ]

        for item in shape_format:
            cpu_input1, _ = create_common_tensor(item[0], 0, 100)
            cpu_input2 = cpu_input1 + 1.j * cpu_input1
            npu_input2 = cpu_input2.npu()
            cpu_output = self.cpu_complex_exec(cpu_input2, item[1])
            npu_output = self.npu_complex_exec(npu_input2, item[1])
            cpu_output = cpu_output.float()
            npu_output = npu_output.float()
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
