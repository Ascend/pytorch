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
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuGeGluBackward(TestCase):

    def cpu_op_exec(self, x_cpu):
        x_cpu_left, x_cpu_right = x_cpu.chunk(2, dim=-1)
        x_cpu_right = x_cpu_right.to(torch.float)
        gelu_cpu = torch.nn.functional.gelu(x_cpu_right)
        gelu_cpu = gelu_cpu.to(torch.float16)
        y_cpu = x_cpu_left * gelu_cpu
        return y_cpu, gelu_cpu

    def npu_op_exec(self, x_npu):
        y_npu, gelu_npu = torch_npu.npu_geglu(x_npu, -1, 1, False)
        return y_npu, gelu_npu

    @unittest.skip("skip test_npu_geglu now")
    def test_npu_geglu_fp16(self):
        data_x = np.random.uniform(-1, 1, [2, 10, 1024]).astype(np.float16)

        x_cpu = torch.from_numpy(data_x)
        y_cpu, gelu_cpu = self.cpu_op_exec(x_cpu)

        x_npu = torch.from_numpy(data_x).npu()
        y_npu, gelu_npu = self.npu_op_exec(x_npu)

        self.assertRtolEqual(y_cpu.numpy(), y_npu.cpu().numpy())
        self.assertRtolEqual(gelu_cpu.numpy(), gelu_npu.cpu().numpy())

    @unittest.skip("skip test_npu_geglu now")
    def test_npu_geglu_fp32(self):
        data_x = np.random.uniform(-1, 1, [2, 10, 1024]).astype(np.float)

        x_cpu = torch.from_numpy(data_x)
        y_cpu, gelu_cpu = self.cpu_op_exec(x_cpu)

        x_npu = torch.from_numpy(data_x).npu()
        y_npu, gelu_npu = self.npu_op_exec(x_npu)

        self.assertRtolEqual(y_cpu.numpy(), y_npu.cpu().numpy())
        self.assertRtolEqual(gelu_cpu.numpy(), gelu_npu.cpu().numpy())


if __name__ == "__main__":
    run_tests()
