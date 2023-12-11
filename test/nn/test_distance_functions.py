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
import torch.nn as nn
import torch.nn.functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestDistanceFunctions(TestCase):
    def test_CosineSimilarity(self):
        input1 = torch.randn(100, 128).npu()
        input2 = torch.randn(100, 128).npu()
        cos = nn.CosineSimilarity(dim=1, eps=1e-6).npu()
        output = cos(input1, input2)
        self.assertEqual(output is not None, True)

    def test_PairwiseDistance(self):
        pdist = nn.PairwiseDistance(p=2).npu()
        input1 = torch.randn(100, 128).npu()
        input2 = torch.randn(100, 128).npu()
        output = pdist(input1, input2)
        self.assertEqual(output is not None, True)

    def test_pairwise_distance(self):
        input1 = torch.randn(2, 3)
        input2 = torch.randn(2, 3)
        npu_input1 = copy.deepcopy(input1).npu()
        npu_input2 = copy.deepcopy(input2).npu()

        cpu_output = F.pairwise_distance(input1, input2)
        npu_output = F.pairwise_distance(npu_input1, npu_input2)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())

    def test_cosine_similarity(self):
        input1 = torch.randn(2, 3)
        input2 = torch.randn(2, 3)
        npu_input1 = copy.deepcopy(input1).npu()
        npu_input2 = copy.deepcopy(input2).npu()

        cpu_output = F.cosine_similarity(input1, input2)
        npu_output = F.cosine_similarity(npu_input1, npu_input2)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())

    def test_pdist(self):
        input1 = torch.randn(2, 3)
        npu_input = copy.deepcopy(input1).npu()

        cpu_output = F.pdist(input1)
        npu_output = F.pdist(npu_input)

        self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().cpu().numpy())


if __name__ == "__main__":
    run_tests()
