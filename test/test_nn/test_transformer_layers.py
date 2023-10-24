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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestTransformerLayers(TestCase):
    def test_Transformer(self):
        transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12).npu()
        src = torch.rand((10, 32, 512)).npu()
        tgt = torch.rand((20, 32, 512)).npu()
        out = transformer_model(src, tgt)
        self.assertEqual(out is not None, True)

    def test_TransformerEncoder(self):
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8).npu()
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6).npu()
        src = torch.rand(10, 32, 512).npu()
        out = transformer_encoder(src)
        self.assertEqual(out is not None, True)

    def test_TransformerEncoderLayer(self):
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8).npu()
        src = torch.rand(10, 32, 512).npu()
        tgt = torch.rand((20, 32, 512)).npu()
        out = encoder_layer(src)
        self.assertEqual(out is not None, True)

    def test_TransformerDecoder(self):
        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8).npu()
        transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        memory = torch.rand(10, 32, 512).npu()
        tgt = torch.rand(20, 32, 512).npu()
        out = transformer_decoder(tgt, memory)
        self.assertEqual(out is not None, True)

    def test_TransformerDecoderLayer(self):
        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8).npu()
        memory = torch.rand(10, 32, 512).npu()
        tgt = torch.rand(20, 32, 512).npu()
        out = decoder_layer(tgt, memory)
        self.assertEqual(out is not None, True)


if __name__ == "__main__":
    run_tests()
