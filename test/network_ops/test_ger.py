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


class TestGer(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = torch.ger(input1, input2)
        output = output.numpy()

        return output

    def npu_op_exec(self, input1, input2):
        output = torch.ger(input1, input2)
        output = output.to("cpu").numpy()

        return output

    def npu_op_exec_out(self, input1, input2, output):
        torch.ger(input1, input2, out=output)
        output = output.to("cpu").numpy()

        return output

    def ger_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -100, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            if cpu_input2.dtype == torch.float16:
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def ger_out_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -100, 100)
            cpu_input3, npu_input3 = create_common_tensor(item[2], -100, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            if cpu_input2.dtype == torch.float16:
                cpu_input2 = cpu_input2.to(torch.float32)
            if cpu_input3.dtype == torch.float16:
                cpu_input3 = cpu_input3.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output_out = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3)
            cpu_output = cpu_output.astype(npu_output_out.dtype)
            self.assertRtolEqual(cpu_output, npu_output_out)

    def test_ger_result(self, device="npu"):
        shape_format = [
            [[np.float16, 0, [128]], [np.float16, 0, [256]]],
            [[np.float16, 0, [128]], [np.float16, 0, [58]]],
            [[np.float16, 0, [128]], [np.float16, 0, [3]]],
            [[np.float16, 0, [128]], [np.float16, 0, [116]]],
            [[np.float32, 0, [256]], [np.float32, 0, [128]]],
            [[np.float32, 0, [256]], [np.float32, 0, [3]]],
            [[np.float32, 0, [2]], [np.float32, 0, [3]]],
            [[np.float32, 0, [128]], [np.float32, 0, [232]]],
        ]
        self.ger_result(shape_format)

    def test_ger_out_result(self, device="npu"):
        shape_format = [
            [[np.float16, 0, [128]], [np.float16, 0, [256]], [np.float16, 0, [256, 116]]],
            [[np.float16, 0, [128]], [np.float16, 0, [58]], [np.float16, 0, [58, 58, 1, 1]]],
            [[np.float16, 0, [128]], [np.float16, 0, [3]], [np.float16, 0, [3, 3]]],
            [[np.float16, 0, [128]], [np.float16, 0, [116]], [np.float16, 0, [128, 116]]],
            [[np.float32, 0, [256]], [np.float32, 0, [128]], [np.float32, 0, [128, 128, 3, 3]]],
            [[np.float32, 0, [256]], [np.float32, 0, [3]], [np.float32, 0, [256, 3]]],
            [[np.float32, 0, [2]], [np.float32, 0, [3]], [np.float32, 0, [3, 1, 3, 3]]],
            [[np.float32, 0, [128]], [np.float32, 0, [232]], [np.float32, 0, [232, 232]]],
        ]
        self.ger_out_result(shape_format)


if __name__ == "__main__":
    run_tests()
