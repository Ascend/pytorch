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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from torch_npu.contrib.function import fuse_add_softmax_dropout

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestFuseAddSoftmaxDropout(unittest.TestCase):
    def test_fuse_add_softmax_dropout(seif):
        training = True
        dropout = nn.DropoutWithByteMask(0.1)
        npu_input1 = torch.rand(96, 12, 384, 384).half().npu()
        npu_input2 = torch.rand(96, 12, 384, 384).half().npu()
        alpha = 0.125
        axis = -1
        
        output = fuse_add_softmax_dropout(training=training, dropout=dropout, \
                                          attn_mask=npu_input1, attn_scores=npu_input2, 
                                          attn_head_size=alpha, p=axis)

if __name__ == "__main__":
    run_tests()

