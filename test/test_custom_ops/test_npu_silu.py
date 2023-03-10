# Copyright (c) 2023 Huawei Technologies Co., Ltd
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


class TestNpuSilu(TestCase):

    def cpu_op_exec_silu(self, input1):
        output = input1 * torch.nn.functional.sigmoid(input1)
        output = output.cpu().numpy()
        return output

    def cpu_op_exec_silu_(self, input1):
        result = input1 * torch.nn.functional.sigmoid(input1)
        input1 = result.cpu().numpy()
        return input1

    def npu_op_exec_silu(self, input1):
        output = torch_npu.npu_silu(input1)
        output = output.cpu().numpy()
        return output

    def npu_op_exec_silu_(self, input1):
        torch_npu.npu_silu_(input1)
        return input1.cpu().numpy()

    def test_silu(self):
        input1 = torch.randn(5, 5).npu()
        cput_out = self.cpu_op_exec_silu(input1)
        npu_out = self.npu_op_exec_silu(input1)
        self.assertRtolEqual(cput_out, npu_out)

    def test_silu_(self):
        input1 = torch.randn(5, 5).npu()
        input2 = torch.clone(input1).npu()
        input1 = self.cpu_op_exec_silu_(input1)
        input2 = self.npu_op_exec_silu_(input2)
        self.assertRtolEqual(input1, input2)


if __name__ == "__main__":
    run_tests()
