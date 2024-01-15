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

    def cpu_op_exec(self, x_cpu, dy_cpu):
        x_cpu_left, x_cpu_right = x_cpu.chunk(2, dim=-1)
        x_cpu_right = x_cpu_right.to(torch.float)
        gelu_cpu = torch.nn.functional.gelu(x_cpu_right)
        gelu_cpu = gelu_cpu.to(torch.float16)
        y_cpu = x_cpu_left * gelu_cpu
        y_cpu.backward(dy_cpu)
        dx_cpu = x_cpu.grad
        return dx_cpu

    def npu_op_exec(self, x_npu, dy_npu):
        y_npu, gelu_npu = torch_npu.npu_geglu(x_npu, -1, 1, False)
        y_npu.backward(dy_npu)
        dx_npu = x_npu.grad
        return dx_npu

    @unittest.skip("skip test_npu_geglu_backward now")
    def test_npu_geglu_backward_fp16(self):
        data_x = np.random.uniform(-1, 1, [2, 10, 1024]).astype(np.float16)
        data_dy = np.random.uniform(-1, 1, [2, 10, 512]).astype(np.float16)

        x_cpu = torch.from_numpy(data_x)
        x_cpu.requires_grad = True
        dy_cpu = torch.from_numpy(data_dy)
        dx_cpu = self.cpu_op_exec(x_cpu, dy_cpu)

        x_npu = torch.from_numpy(data_x).npu()
        x_npu.requires_grad = True
        dy_npu = torch.from_numpy(data_dy).npu()
        dx_npu = self.npu_op_exec(x_npu, dy_npu)

        self.assertRtolEqual(dx_cpu.numpy(), dx_npu.cpu().numpy())

    @unittest.skip("skip test_npu_geglu_backward now")
    def test_npu_geglu_backward_fp32(self):
        data_x = np.random.uniform(-1, 1, [2, 10, 1024]).astype(np.float)
        data_dy = np.random.uniform(-1, 1, [2, 10, 512]).astype(np.float)

        x_cpu = torch.from_numpy(data_x)
        x_cpu.requires_grad = True
        dy_cpu = torch.from_numpy(data_dy)
        dx_cpu = self.cpu_op_exec(x_cpu, dy_cpu)

        x_npu = torch.from_numpy(data_x).npu()
        x_npu.requires_grad = True
        dy_npu = torch.from_numpy(data_dy).npu()
        dx_npu = self.npu_op_exec(x_npu, dy_npu)

        self.assertRtolEqual(dx_cpu.numpy(), dx_npu.cpu().numpy())


if __name__ == "__main__":
    run_tests()
