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
import torch_npu
from torch_npu.contrib.module import MultiheadAttention
from torch_npu.contrib.module.multihead_attention import MHAConfig
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor

FORMAT_ND = 2
FORMAT_NZ = 29
npu_device = "npu:0"
MHAConfig.set_fussion()


class TestMultiheadAttention(unittest.TestCase):
    def test_MultiheadAttention(self):
        model = MultiheadAttention(embed_dim=1024,
                                   num_heads=16,
                                   dropout=0.1,
                                   kdim=1024,
                                   vdim=1024,
                                   self_attention=True,
                                   encoder_decoder_attention=True)
        _, query = create_common_tensor([np.float16, FORMAT_NZ, (1024, 1024)], -1, 1)
        _, key = create_common_tensor([np.float16, FORMAT_NZ, (1024, 1024)], -1, 1)
        _, value = create_common_tensor([np.float16, FORMAT_NZ, (1024, 1024)], -1, 1)
        _, key_padding_mask = create_common_tensor([np.float16, FORMAT_NZ, (16, 16, 64, 64)], -65504, 65504)
        bsz = 16
        tgt_len = 64
        s_len = 64
        model = model.to("npu")
        output = model(query, key, value, bsz, tgt_len, s_len, key_padding_mask)


if __name__ == "__main__":
    run_tests()
