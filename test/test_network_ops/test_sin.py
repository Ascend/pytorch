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


class TestSin(TestCase):
    def cpu_op_exec(self, input1):
        output = torch.sin(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.sin(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2):
        torch.sin(input1, out=input2)
        output = input2.to("cpu")
        output = output.numpy()
        return output

    def test_sin_common_shape_format(self, device="npu"):
        shape_format = [
            [[np.float32, 0, (5, 3)]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -10, 10)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_sin_out_common_shape_format(self, device="npu"):
        shape_format = [
            [[np.float16, -1, (4, 3, 128, 128)], [np.float16, -1, (4, 3, 128, 128)]],
            [[np.float16, 0, (4, 3, 128, 128)], [np.float16, 0, (10, 3, 64, 128)]],
            [[np.float16, 0, (4, 3, 128, 128)], [np.float16, 0, (2, 3, 256, 128)]],
            [[np.float32, 0, (4, 3, 128, 128)], [np.float32, 0, (4, 3, 128, 128)]],
            [[np.float32, 0, (4, 3, 128, 128)], [np.float32, 0, (8, 3, 64, 128)]],
            [[np.float32, -1, (4, 3, 128, 128)], [np.float32, -1, (4, 3, 256, 64)]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -10, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[0], -10, 10)
            cpu_input3, npu_input3 = create_common_tensor(item[1], -10, 10)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output_out1 = self.npu_op_exec_out(npu_input1, npu_input2)
            npu_output_out2 = self.npu_op_exec_out(npu_input1, npu_input3)
            cpu_output = cpu_output.astype(npu_output_out1.dtype)
            self.assertRtolEqual(cpu_output, npu_output_out1)
            self.assertRtolEqual(cpu_output, npu_output_out2)


if __name__ == "__main__":
    run_tests()
