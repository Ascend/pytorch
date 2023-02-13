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


class TestNpuMish(TestCase):

    def cpu_op_exec(self, input1):
        output = input1 * \
            torch.nn.functional.tanh(torch.nn.functional.softplus(input1))
        output = output.cpu().numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch_npu.npu_mish(input1)
        output = output.cpu().numpy()
        return output

    def test_mish(self):
        input1 = torch.randn(5, 5).npu()
        cpu_out = self.cpu_op_exec(input1)
        npu_out = self.npu_op_exec(input1)
        self.assertRtolEqual(cpu_out, npu_out)


if __name__ == "__main__":
    run_tests()
