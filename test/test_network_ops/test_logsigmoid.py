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

# coding: utf-8

import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestLogsigmoid(TestCase):

    def cpu_op_exec(self, input1):
        output = torch.nn.functional.logsigmoid(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.nn.functional.logsigmoid(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_sigmoid_shape_format(self, device="npu"):
        shape_format = [
            [[np.float32, 0, (6, 4)]],
            [[np.float32, 3, (2, 4, 5)]],
            [[np.float32, 4, (1, 2, 3, 3)]],
            [[np.float32, 29, (1, 2, 3, 3)]]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_sigmoid_shape_format_float16(self, device="npu"):
        shape_format1 = [
            [[np.float16, 0, (6, 4)]],
            [[np.float16, 3, (2, 4, 5)]],
            [[np.float16, 4, (1, 2, 3, 3)]],
            [[np.float16, 29, (1, 2, 3, 3)]]
        ]
        for item in shape_format1:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output1 = self.cpu_op_exec(cpu_input1)
            npu_output1 = self.npu_op_exec(npu_input1)
            cpu_output1 = cpu_output1.astype(npu_output1.dtype)
            self.assertRtolEqual(cpu_output1, npu_output1)


if __name__ == "__main__":
    run_tests()
