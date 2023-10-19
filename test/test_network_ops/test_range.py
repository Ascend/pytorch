# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION.
# All rights reserved.
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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestRange(TestCase):

    def cpu_op_exec(self, start, limit, dtype, dev):
        output = torch.range(start, limit, dtype=dtype, device=dev)
        output = output.numpy()
        return output

    def npu_op_exec(self, start, limit, dtype, dev):
        output = torch.range(start, limit, dtype=dtype, device=dev)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_step_exec(self, start, limit, delta, dtype, dev):
        output = torch.range(start, limit, delta, dtype=dtype, device=dev)
        output = output.numpy()
        return output

    def npu_op_step_exec(self, start, limit, delta, dtype, dev):
        output = torch.range(start, limit, delta, dtype=dtype, device=dev)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_out_exec(self, start, limit, delta, dtype, output):
        torch.range(start, limit, delta, dtype=dtype, out=output)
        output = output.numpy()
        return output

    def npu_op_out_exec(self, start, limit, delta, dtype, output):
        torch.range(start, limit, delta, dtype=dtype, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_range(self, device="npu"):
        shape_format = [
            [-10, 10, torch.float32],
            [50, 100, torch.int32],
            [1, 100, torch.float32],
            [0, 100, torch.float32],
        ]

        for item in shape_format:
            cpu_output = self.cpu_op_exec(item[0], item[1], item[2], 'cpu')
            npu_output = self.npu_op_exec(item[0], item[1], item[2], 'npu')
            self.assertRtolEqual(cpu_output, npu_output)

    def test_range_step(self, device="npu"):
        shape_format = [
            [-10, 10, 0.5, torch.float32],
            [1, 100, 1, torch.int32],
            [100, 0, -2, torch.float32],
            [0, -100, -2, torch.float32],
        ]

        for item in shape_format:
            cpu_output = self.cpu_op_step_exec(item[0], item[1], item[2], item[3], 'cpu')
            npu_output = self.npu_op_step_exec(item[0], item[1], item[2], item[3], 'npu')
            self.assertRtolEqual(cpu_output, npu_output)

    def test_range_out_step(self, device="npu"):
        shape_format = [
            [-10, 10, 0.5, torch.float32],
            [1, 100, 1, torch.int32],
            [100, 0, -2, torch.float32],
            [0, -100, -2, torch.float32],
        ]

        for item in shape_format:
            cpu_output = torch.zeros([int((item[1] - item[0]) / item[2] + 1)], dtype=item[3])
            npu_output = cpu_output.to('npu')
            cpu_output = self.cpu_op_out_exec(item[0], item[1], item[2], item[3], cpu_output)
            npu_output = self.npu_op_out_exec(item[0], item[1], item[2], item[3], npu_output)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_range_shape(self):
        a = torch.tensor([[-100, 100]])
        step = 3
        cout = torch.range(a[0][0], a[0][1], step, dtype=torch.float, device="cpu")
        nout = torch.range(a.npu()[0][0], a[0][1], step, dtype=torch.float, device="npu")
        print("cout.shape:", cout.shape)
        print("nout.shape:", nout.shape)


if __name__ == "__main__":
    run_tests()
