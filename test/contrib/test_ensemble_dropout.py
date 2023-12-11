# Copyright (c) 2020 Huawei Technologies Co., Ltd
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

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_npu.contrib.module import NpuFairseqDropout, NpuCachedDropout
from torch_npu.testing.testcase import run_tests


class NpuMNIST(nn.Module):

    def __init__(self):
        super(NpuMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x, dropout):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = dropout(x)
        x = x.view(-1, 40)
        return x


class TestEnsembleDropout(unittest.TestCase):
    def test_EnsembleDropout(self):
        model = NpuMNIST().to("npu")
        x = torch.randn(2, 10, 16, 16).to("npu")
        NpuFairseqDropout.enable_dropout_ensemble(model)
        dropout = NpuFairseqDropout(p=1)
        output = model(x, dropout)

        NpuCachedDropout.enable_dropout_ensemble(model)
        dropout = NpuCachedDropout(p=1)
        output = model(x, dropout)


if __name__ == "__main__":
    run_tests()
