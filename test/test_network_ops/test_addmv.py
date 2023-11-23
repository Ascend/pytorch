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


class TestAddmv(TestCase):
    def cpu_op_exec(self, a, b, c, alpha, beta):
        output = torch.addmv(c, a, b, alpha=alpha, beta=beta)
        output = output.numpy()
        return output

    def npu_op_exec(self, a, b, c, alpha, beta):
        output = torch.addmv(c, a, b, alpha=alpha, beta=beta)
        output = output.to('cpu')
        output = output.numpy()
        return output

    def npu_op_exec_out(self, a, b, c, beta, alpha, input1):
        torch.addmv(c, a, b, alpha=alpha, beta=beta, out=input1)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def test_addmv_fp16(self, device="npu"):
        shape_format = [
            [[np.float16, 3, (2, 3)], [np.float16, 3, (3,)], [np.float16, 3, (2,)]]

        ]
        for item in shape_format:
            input_a, npu_input_a = create_common_tensor(item[0], -2, 2)
            input_b, npu_input_b = create_common_tensor(item[1], -2, 2)
            input_c, npu_input_c = create_common_tensor(item[2], -2, 2)

            input_a = input_a.to(torch.float32)
            input_b = input_b.to(torch.float32)
            input_c = input_c.to(torch.float32)

            cpu_output = self.cpu_op_exec(input_a, input_b, input_c, 1, 1)
            npu_output = self.npu_op_exec(npu_input_a, npu_input_b, npu_input_c, 1, 1)

            cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_addmv_out_fp16(self, device="npu"):
        shape_format = [
            [[np.float16, 3, (2, 3)], [np.float16, 3, (3,)], [np.float16, 3, (2,)], [np.float16, 3, (10,)]]

        ]
        for item in shape_format:
            input_a, npu_input_a = create_common_tensor(item[0], -2, 2)
            input_b, npu_input_b = create_common_tensor(item[1], -2, 2)
            input_c, npu_input_c = create_common_tensor(item[2], -2, 2)
            _, npu_input = create_common_tensor(item[3], -2, 2)

            input_a = input_a.to(torch.float32)
            input_b = input_b.to(torch.float32)
            input_c = input_c.to(torch.float32)

            cpu_output = self.cpu_op_exec(input_a, input_b, input_c, 1, 1)
            npu_output = self.npu_op_exec_out(npu_input_a, npu_input_b, npu_input_c, 1, 1, npu_input)
            cpu_output = cpu_output.astype(np.float16)

            self.assertRtolEqual(cpu_output, npu_output)

    @unittest.skip("skip test_addmv_fp32 now")
    def test_addmv_fp32(self, device="npu"):
        shape_format = [
            [[np.float32, 0, (2, 3)], [np.float32, 0, (3,)], [np.float32, 0, (2,)]],
            [[np.float32, 0, (3168, 320)], [np.float32, 0, (320,)], [np.float32, 0, (3168,)]],
        ]
        for item in shape_format:
            input_a, npu_input_a = create_common_tensor(item[0], -2, 2)
            input_b, npu_input_b = create_common_tensor(item[1], -2, 2)
            input_c, npu_input_c = create_common_tensor(item[2], -2, 2)

            cpu_output = self.cpu_op_exec(input_a, input_b, input_c, 1, 1)
            npu_output = self.npu_op_exec(npu_input_a, npu_input_b, npu_input_c, 1, 1)

            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
