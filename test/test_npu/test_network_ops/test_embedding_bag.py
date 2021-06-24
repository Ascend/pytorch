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
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor
import torch.nn.functional as F

class TestEmbeddingBag(TestCase):
    def test_embedding_bag_1d(self, device):
        cpu_weight = torch.rand(10, 3)
        cpu_indices = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9])
        cpu_offsets = torch.tensor([0, 4])
        npu_weight = cpu_weight.npu()
        npu_indices = cpu_indices.npu()
        npu_offsets = cpu_offsets.npu()
        cpu_output = F.embedding_bag(cpu_weight, cpu_indices, cpu_offsets).detach().numpy()
        npu_output = F.embedding_bag(npu_weight, npu_indices, npu_offsets).cpu().detach().numpy()
        self.assertRtolEqual(cpu_output, npu_output)

    def test_embedding_bag_2d(self, device):
        cpu_weight = torch.rand(10, 3)
        cpu_indices = torch.tensor([[1, 2, 4, 5, 4, 3, 2, 9], [1, 2, 4, 5, 4, 3, 2, 9]])
        npu_weight = cpu_weight.npu()
        npu_indices = cpu_indices.npu()
        cpu_output = F.embedding_bag(cpu_weight, cpu_indices).detach().numpy()
        npu_output = F.embedding_bag(npu_weight, npu_indices).cpu().detach().numpy()
        self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestEmbeddingBag, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()