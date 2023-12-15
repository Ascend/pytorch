# Copyright (c) 2023, Huawei Technologies.All rights reserved.
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


class TestSqrt(TestCase):

    def cpu_op_exec(self, input1):
        output = torch._foreach_sqrt(input1)
        return output

    def npu_op_exec(self, input1):
        output = torch._foreach_sqrt(input1)
        return output

    def cpu_op_exec_(self, input1):
        torch._foreach_sqrt(input1)
        return input1

    def npu_op_exec_(self, input1):
        torch._foreach_sqrt(input1)
        return input1

    @unittest.skip("skip test_sqrt_shape_format now")
    def test_sqrt_shape_format(self):
        shape_format = [
            [[np.float32, 0, (1, 6, 4)]],
            [[np.float32, 3, (2, 4, 5)]]
        ]
        cpu_input1, npu_input1 = create_common_tensor(shape_format[0][0], 1, 100)
        cpu_input2, npu_input2 = create_common_tensor(shape_format[1][0], 1, 100)
        cpu_output = self.cpu_op_exec([cpu_input1, cpu_input2])
        npu_output = self.npu_op_exec([npu_input1, npu_input2])
        for (cpu_tmp1, npu_tmp1) in zip(cpu_output, npu_output):
            self.assertRtolEqual(cpu_tmp1.numpy(), npu_tmp1.to("cpu").numpy())

    @unittest.skip("skip test_sqrt1_shape_format now")
    def test_sqrt1_shape_format(self):
        shape_format = [
            [[np.float32, 0, (1, 6, 4)]],
            [[np.float32, 3, (2, 4, 5)]]
        ]
        cpu_input1, npu_input1 = create_common_tensor(shape_format[0][0], 1, 100)
        cpu_input2, npu_input2 = create_common_tensor(shape_format[1][0], 1, 100)
        cpu_output = self.cpu_op_exec_([cpu_input1, cpu_input2])
        npu_output = self.npu_op_exec_([npu_input1, npu_input2])
        for (cpu_tmp1, npu_tmp1) in zip(cpu_output, npu_output):
            self.assertRtolEqual(cpu_tmp1.numpy(), npu_tmp1.to("cpu").numpy())


if __name__ == "__main__":
    run_tests()
