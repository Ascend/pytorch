# Copyright (c) 2022 Huawei Technologies Co., Ltd
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
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from torch_npu.contrib.function import fuse_add_softmax_dropout

from torch_npu.testing.testcase import TestCase, run_tests


class TestFuseAddSoftmaxDropout(TestCase):
    def npu_fuse_add_softmax_dropout(self, dropout, attn_mask, attn_scores, attn_head_size):
        attn_scores = torch.add(attn_mask, attn_scores, alpha=(1 / math.sqrt(attn_head_size)))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = dropout(attn_probs)
        return attn_probs

    def test_fuse_add_softmax_dropout(self):
        training = True
        dropout = nn.DropoutWithByteMask(0)
        npu_input1 = torch.rand(96, 12, 384, 384).npu().half()
        npu_input2 = torch.rand(96, 12, 384, 384).npu().half()
        alpha = 64
        axis = 0

        npu_output = self.npu_fuse_add_softmax_dropout(dropout, npu_input1, npu_input2, alpha)
        high_performance_output = fuse_add_softmax_dropout(training=training, dropout=dropout,
                                                           attn_mask=npu_input1, attn_scores=npu_input2,
                                                           attn_head_size=alpha, p=axis)

        self.assertRtolEqual(npu_output.detach().cpu().numpy(), high_performance_output.detach().cpu().numpy())


if __name__ == "__main__":
    run_tests()
