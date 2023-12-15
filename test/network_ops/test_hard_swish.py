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


class TestHardSwish(TestCase):
    def cpu_op_exec(self, input1):
        cpu_output = torch.nn.functional.hardswish(input1, inplace=False)
        cpu_output = cpu_output.numpy()

        return cpu_output

    def npu_op_exec(self, input1):
        output = torch.nn.functional.hardswish(input1, inplace=False)
        output = output.to("cpu")
        output = output.numpy()

        return output

    def create_shape_format16(self):
        format_list = [0, 3, 29]
        dtype_list = [np.float16]
        shape_list = [[32], [32, 3], [32, 3, 3], [64, 32, 3, 3]]
        shape_format = [[i, j, k] for i in dtype_list for j in format_list for k in shape_list]

        return shape_format

    def create_shape_format32(self):
        format_list32 = [0, 3, 29]
        dtype_list32 = [np.float32]
        shape_list32 = [[32], [32, 3], [32, 3, 3], [64, 32, 3, 3]]
        shape_format32 = [[i, j, k] for i in dtype_list32 for j in format_list32 for k in shape_list32]

        return shape_format32

    def test_hardswish_shape_format_fp16(self):
        for item in self.create_shape_format16():
            cpu_input1, npu_input1 = create_common_tensor(item, 2, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)

            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            cpu_output = cpu_output.astype(npu_output.dtype)

            self.assertRtolEqual(cpu_output, npu_output)

    def test_hardswish_shape_format_fp32(self):
        for item in self.create_shape_format32():
            cpu_input1, npu_input1 = create_common_tensor(item, 2, 100)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)

            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
