# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION.
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

import copy
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestGru(TestCase):
    def test_gru(self, device="npu"):
        shape_format = [
            [[np.float16, (1, 3, 2)], [np.float16, (1, 3, 2)], 2, 2, 1, False, True, False],
            [[np.float32, (2, 1, 1)], [np.float32, (1, 2, 2)], 1, 2, 1, False, False, True],
            [[np.float16, (1, 3, 1)], [np.float16, (2, 3, 2)], 1, 2, 2, False, True, False],
            [[np.float32, (1, 1, 2)], [np.float32, (1, 1, 3)], 2, 3, 1, False, False, False],
            [[np.float16, (1, 1, 1)], [np.float16, (3, 1, 1)], 1, 1, 3, False, True, True],
        ]

        for item in shape_format:
            cpu_gru = torch.nn.GRU(input_size=item[2], hidden_size=item[3], num_layers=item[4],
                                   bidirectional=item[5], bias=item[-2], batch_first=item[-1])
            npu_gru = copy.deepcopy(cpu_gru).npu()

            input1 = np.random.uniform(0, 1, item[0][1]).astype(item[0][0])
            if item[0][0] == np.float16:
                cpu_input1 = torch.from_numpy(input1.astype(np.float32))
            else:
                cpu_input1 = torch.from_numpy(input1)
            npu_input1 = torch.from_numpy(input1).npu()

            h0 = np.random.uniform(0, 1, item[1][1]).astype(item[1][0])
            if item[1][0] == np.float16:
                cpu_h0 = torch.from_numpy(h0.astype(np.float32))
            else:
                cpu_h0 = torch.from_numpy(h0)
            npu_h0 = torch.from_numpy(h0).npu()

            cpu_output_y, cpu_output_h = cpu_gru(cpu_input1, cpu_h0)
            npu_output_y, npu_output_h = npu_gru(npu_input1, npu_h0)

            if item[0][0] == np.float16:
                self.assertRtolEqual(cpu_output_y.detach().numpy().astype(np.float16),
                                     npu_output_y.cpu().detach().numpy())
                self.assertRtolEqual(cpu_output_h.detach().numpy().astype(np.float16),
                                     npu_output_h.cpu().detach().numpy())
            else:
                # Ascend: fp33 isn't enough precision, relaxation of precision requirement temporary
                self.assertRtolEqual(cpu_output_y.detach().numpy(), npu_output_y.cpu().detach().numpy(), prec=1.e-1)
                self.assertRtolEqual(cpu_output_h.detach().numpy(), npu_output_h.cpu().detach().numpy(), prec=1.e-1)

    def pad_seq(self, pad_token, seq, seq_len, max_length):
        seq += [pad_token for _ in range(max_length - seq_len)]
        return seq

    def test_pack_gru_pad(self):
        batch_size = 3
        max_len = 6
        embedding_size = 8
        hidden_size = 16
        vocab_size = 20
        input_seq = [[3, 5, 12, 7, 2], [4, 11, 14], [18, 7, 3, 8, 5, 4]]
        lengths = [5, 3, 6]

        embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        GRU_cpu = torch.nn.GRU(embedding_size, hidden_size, batch_first=True)
        GRU_npu = copy.deepcopy(GRU_cpu).npu()

        input_seq = sorted(input_seq, key=lambda tp: len(tp), reverse=True)
        lengths = sorted(lengths, key=lambda tp: tp, reverse=True)
        pad_token = 0
        pad_seqs = []
        for i, j in zip(input_seq, lengths):
            pad_seqs.append(self.pad_seq(pad_token, i, j, max_len))

        pad_seqs = torch.tensor(pad_seqs)
        embeded = embedding(pad_seqs)

        pack_cpu = torch.nn.utils.rnn.pack_padded_sequence(embeded, lengths, batch_first=True)
        pack_npu = torch.nn.utils.rnn.pack_padded_sequence(embeded.npu(), lengths, batch_first=True)

        state = None
        gru_cpu, _ = GRU_cpu(pack_cpu, state)
        gru_npu, _ = GRU_npu(pack_npu, state)

        pad_cpu, others = torch.nn.utils.rnn.pad_packed_sequence(gru_cpu, batch_first=True)
        pad_npu, others = torch.nn.utils.rnn.pad_packed_sequence(gru_npu, batch_first=True)

        self.assertRtolEqual(pad_cpu.detach().numpy(), pad_npu.cpu().detach().numpy(), prec=1.e-3)


if __name__ == "__main__":
    run_tests()
