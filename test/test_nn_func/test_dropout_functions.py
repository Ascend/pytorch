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

import copy
import torch
import torch.nn.functional as F
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestDropoutFunctions(TestCase):
    def test_dropout(self):
        input1 = torch.randn(10, 30)
        npu_input = copy.deepcopy(input1).npu()
        cpu_output = F.dropout(input1)
        npu_output = F.dropout(npu_input)

        self.assertTrue(str(cpu_output.shape), str(npu_output.shape))

    def test_alpha_dropout(self):
        input1 = torch.randn(10, 30)
        npu_input = copy.deepcopy(input1).npu()
        cpu_output = F.alpha_dropout(input1)
        npu_output = F.alpha_dropout(npu_input)

        self.assertTrue(str(cpu_output.shape), str(npu_output.shape))

    def test_dropout2d(self):
        input1 = torch.randn(10, 30)
        npu_input = copy.deepcopy(input1).npu()
        cpu_output = F.dropout2d(input1)
        npu_output = F.dropout2d(npu_input)

        self.assertTrue(str(cpu_output.shape), str(npu_output.shape))

    def test_dropout3d(self):
        input1 = torch.randn(10, 30)
        npu_input = copy.deepcopy(input1).npu()
        cpu_output = F.dropout3d(input1)
        npu_output = F.dropout3d(npu_input)

        self.assertTrue(str(cpu_output.shape), str(npu_output.shape))


if __name__ == "__main__":
    run_tests()
