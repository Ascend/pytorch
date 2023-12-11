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
import torch.nn as nn
import torch.nn.functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestSparseFunctions(TestCase):
    def test_embedding(self):
        input1 = torch.tensor([[0, 1, 1, 2], [3, 5, 7, 11]], dtype=torch.long)
        embd = nn.Embedding(20, 20)
        weight = embd.weight
        npu_input = input1.npu().int()
        npu_weight = weight.npu()
        cpu_output = F.embedding(input1, weight)
        npu_output = F.embedding(npu_input, npu_weight)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())

    def test_one_hot(self):
        input1 = torch.arange(0, 5) % 3
        npu_input = input1.npu().int()

        cpu_output = F.one_hot(input1)
        npu_output = F.one_hot(npu_input)

        self.assertRtolEqual(cpu_output.detach().int().numpy(), npu_output.detach().cpu().numpy())


if __name__ == "__main__":
    run_tests()
