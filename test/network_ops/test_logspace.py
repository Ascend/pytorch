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


class TestLogSpace(TestCase):
    def cpu_op_exec(self, start, end, steps, base):
        output = torch.logspace(start=start, end=end, steps=steps, base=base)
        output = output.numpy()
        return output

    def npu_op_exec(self, start, end, steps, base):
        output = torch.logspace(start=start, end=end, steps=steps, base=base, device="npu")
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, start, end, steps, base, dtype, output):
        torch.logspace(start=start, end=end, steps=steps, base=base, dtype=dtype, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_fp16(self, start, end, steps, base, dtype):
        output = torch.logspace(start=start, end=end, steps=steps, base=base, dtype=torch.float32)
        output = output.numpy()
        output = output.astype(np.float16)
        return output

    def npu_op_exec_dtype(self, start, end, steps, base, dtype):
        output = torch.logspace(start=start, end=end, steps=steps, base=base, dtype=dtype, device="npu")
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_logspace_common_shape_format(self, device="npu"):
        shape_format = [
            [0.0, 1.0, 10, 0.2, torch.float32],
            [2.0, 3.0, 10, 0.05, torch.float32],
            [10.0, 10.5, 11, 0.2, torch.float32],
            [10.0, 10.5, 110, 0.2, torch.float32],
            [0.0, 0.1, 20, 1.2, torch.float32],
            [0.5, 1.0, 50, 8.0, torch.float32],
            [1.0, 2.0, 2, 0.5, torch.float32],
            [0.0, 0.0, 1, 0.1, torch.float32],
            [1.0, 1.0, 1, 0.1, torch.float32],
            [1.0, 1.0, 0, 0.1, torch.float32],
            [1.0, 2.0, 9, 0.1, torch.float32]
        ]
        for item in shape_format:
            cpu_output = self.cpu_op_exec(item[0], item[1], item[2], item[3])
            npu_output = self.npu_op_exec(item[0], item[1], item[2], item[3])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_logspace_out_common_shape_format(self, device="npu"):
        shape_format = [
            [0.0, 1.0, 10, 0.2, torch.float32, [np.float32, 0, [10, 2]]],
            [2.0, 3.0, 10, 0.05, torch.float32, [np.float32, 0, [10, 2, 5]]],
            [10.0, 10.5, 11, 0.2, torch.float32, [np.float32, 0, [10, 2, 5, 20]]],
            [10.0, 10.5, 110, 0.2, torch.float32, [np.float32, 0, [10]]],
            [0.0, 0.1, 20, 1.2, torch.float32, [np.float32, 0, [10, 20, 5, 20]]],
            [0.0, 1.0, 10, 0.2, torch.float16, [np.float16, 0, [10, 2]]],
            [2.0, 3.0, 10, 0.05, torch.float16, [np.float16, 0, [10, 2, 5]]],
            [10.0, 10.5, 11, 0.2, torch.float16, [np.float16, 0, [10, 2, 5, 20]]],
            [10.0, 10.5, 110, 0.2, torch.float16, [np.float16, 0, [10]]],
            [0.0, 0.1, 20, 1.2, torch.float16, [np.float16, 0, [10, 20, 5, 20]]],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[5], -10, 10)
            cpu_output = self.cpu_op_exec(item[0], item[1], item[2], item[3])
            npu_output = self.npu_op_exec_out(item[0], item[1], item[2], item[3], item[4], npu_input)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    @unittest.skip("skip test_logspace_float16_shape_format now")
    def test_logspace_float16_shape_format(self, device="npu"):
        shape_format = [
            [-2.0, 2.0, 32, 32, torch.float16],
            [0.0, 1.0, 10, 0.2, torch.float16],
            [2.0, 3.0, 10, 0.05, torch.float16],
            [0.0, 0.1, 20, 1.2, torch.float16],
            [0.5, 1.0, 50, 8.0, torch.float16],
            [1.0, 2.0, 2, 0.5, torch.float16],
            [0.0, 0.0, 1, 0.1, torch.float16]
        ]
        for item in shape_format:
            cpu_output = self.cpu_op_exec_fp16(item[0], item[1], item[2], item[3], item[4])
            npu_output = self.npu_op_exec_dtype(item[0], item[1], item[2], item[3], item[4])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
