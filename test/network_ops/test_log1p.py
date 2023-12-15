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


class TestLog1p(TestCase):
    def cpu_op_exec(self, input1):
        output = torch.log1p(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.log1p(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_(self, input1):
        input1.log1p_()
        input1 = input1.numpy()
        return input1

    def npu_op_exec_(self, input1):
        input1.log1p_()
        input1 = input1.to("cpu")
        input1 = input1.numpy()
        return input1

    def cpu_op_exec_fp16(self, input1):
        input1 = input1.to(torch.float32)
        output = torch.log1p(input1)
        output = output.numpy()
        output = output.astype(np.float16)
        return output

    def test_log1p_common_shape_format(self, device="npu"):
        shape_format = [
            [[np.float32, 0, 1]],
            [[np.float32, 0, (64, 10)]],
            [[np.float32, 4, (32, 1, 3, 3)]],
            [[np.float32, 29, (10, 128)]]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], -1, 1)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            cpu_output1 = self.cpu_op_exec_(cpu_input)
            npu_output1 = self.npu_op_exec_(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output1, npu_output1)

    def test_log1p_float16_shape_format(self, device="npu"):
        shape_format = [
            [[np.float16, -1, 1]],
            [[np.float16, -1, (64, 10)]],
            [[np.float16, -1, (31, 1, 3)]]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], -1, 1)
            cpu_output = self.cpu_op_exec_fp16(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
