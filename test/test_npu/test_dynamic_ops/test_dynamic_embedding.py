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
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor
from common_utils import TestCase, run_tests
import time
import os
import copy
# Need export DYNAMIC_COMPILE_ENABLE=1 and export EXPERIMENTAL_DYNAMIC_PARTITION=1

class EmbeddingFuncNet(torch.nn.Module):
    def __init__(self):
        super(EmbeddingFuncNet, self).__init__()

    def forward(self, indices, weight):
        out = torch.nn.functional.embedding(indices, weight)
        return out

class EmbeddingNet(torch.nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()

    def forward(self, indices, embed):
        out =embed(indices)
        return out

class TestShape(TestCase):
    def generate_weight(self, x, y):
        rand_data = np.random.randn(x,y).astype(np.float32)
        cpu_out = torch.from_numpy(rand_data)
        npu_out = torch.from_numpy(rand_data).npu()
        return cpu_out.to(torch.float), npu_out.to(torch.float)

    def generate_indices(self, shape, min, max):
        rand_data = np.random.randint(min, max, shape)
        cpu_out = torch.from_numpy(rand_data)
        npu_out = torch.from_numpy(rand_data).npu()
        return cpu_out.to(torch.long), npu_out.to(torch.long)

    def test_dynamic_threads_support_op(self, device):
        shape_list1 = [[40, 32], [40, 1024], [40000, 1024], [33712, 1024]]
        shape_list2 = [[40], [40,3125], [64, 7, 128]]
        shape_format = [
            [i, j] for i in shape_list1 for j in shape_list2
        ]
        net_func = EmbeddingFuncNet()
        net = EmbeddingNet()
        for item in shape_format:
            weight_cpu, weight_npu = self.generate_weight(item[0][0], item[0][1])
            indices_cpu, indices_npu = self.generate_indices(item[1], 1, item[0][0])
            cpu_out = net_func(indices_cpu, weight_cpu)
            npu_out = net_func(indices_npu, weight_npu)
            npu_output = npu_out.to("cpu")
            self.assertRtolEqual(cpu_out.numpy(), npu_output.numpy())

            embed_cpu = torch.nn.Embedding(item[0][0], item[0][1])
            embed_npu = copy.deepcopy(embed_cpu).npu()
            cpu_out = net(indices_cpu, embed_cpu)
            npu_out = net(indices_npu, embed_npu)
            npu_output = npu_out.to("cpu")
            self.assertRtolEqual(cpu_out.detach().numpy(), npu_output.detach().numpy())


instantiate_device_type_tests(TestShape, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()