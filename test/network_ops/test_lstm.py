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


class TestLstm(TestCase):
    def test_lstm_single_direction(self):
        # shape_format:[[dtype, (num_step, batch_size, input_size)],
        # num_layers, input_size, hidden_size, is_training, batch_first]
        shape_format = [
            [[np.float16, (16, 32, 64)], 1, 64, 32, True, True],
            [[np.float16, (5, 32, 64)], 1, 64, 32, False, True],
            [[np.float32, (5, 32, 64)], 1, 64, 64, True, False],
            [[np.float32, (5, 32, 64)], 1, 64, 64, False, False],
            [[np.float32, (26, 2560, 512)], 1, 512, 256, False, True],
            [[np.float32, (10, 33, 128)], 1, 128, 64, False, False],
            [[np.float16, (16, 32, 64)], 2, 64, 32, True, True],
            [[np.float16, (5, 32, 64)], 2, 64, 32, False, True],
            [[np.float32, (5, 32, 64)], 2, 64, 64, True, False],
            [[np.float32, (5, 32, 64)], 2, 64, 64, False, False],
            [[np.float32, (26, 2560, 512)], 2, 512, 256, False, True],
            [[np.float32, (10, 33, 128)], 2, 128, 64, False, False],
            [[np.float32, (32, 64)], 2, 64, 64, False, False],
            [[np.float32, (2560, 512)], 2, 512, 256, False, True],
            [[np.float32, (33, 128)], 2, 128, 64, False, False],
        ]

        for item in shape_format:
            cpu_lstm = torch.nn.LSTM(input_size=item[2], hidden_size=item[3], batch_first=item[5],
                                     num_layers=item[1], bidirectional=False, bias=False)
            cpu_lstm.training = item[4]
            npu_lstm = copy.deepcopy(cpu_lstm).npu()

            input1 = np.random.uniform(0, 1, item[0][1]).astype(np.float16).astype(np.float32)
            cpu_input1 = torch.from_numpy(input1)
            cpu_output_y, (cpu_output_h, cpu_output_c) = cpu_lstm(cpu_input1)

            npu_input1 = torch.from_numpy(input1.astype(item[0][0])).npu()
            npu_output_y, (npu_output_h, npu_output_c) = npu_lstm(npu_input1)

            self.assertRtolEqual(cpu_output_y.detach().numpy(),
                                 npu_output_y.cpu().to(torch.float).detach().numpy(), prec=1.e-3)
            self.assertRtolEqual(cpu_output_h.detach().numpy(),
                                 npu_output_h.cpu().to(torch.float).detach().numpy(), prec=1.e-3)
            self.assertRtolEqual(cpu_output_c.detach().numpy(),
                                 npu_output_c.cpu().to(torch.float).detach().numpy(), prec=2.e-3)

    def test_lstm_bidirection(self):
        # shape_format:[[dtype, (num_step, batch_size, input_size)],
        # num_layers, input_size, hidden_size, is_training]
        shape_format = [
            [[np.float16, (16, 32, 64)], 1, 64, 32, True], [[np.float16, (5, 32, 64)], 1, 64, 32, False],
            [[np.float32, (5, 32, 64)], 1, 64, 64, True], [[np.float32, (5, 32, 64)], 1, 64, 64, False],
            [[np.float32, (26, 2560, 512)], 1, 512, 256, False], [[np.float32, (10, 33, 128)], 1, 128, 64, False],
            [[np.float16, (16, 32, 64)], 2, 64, 32, True], [[np.float16, (5, 32, 64)], 2, 64, 32, False],
            [[np.float32, (5, 32, 64)], 2, 64, 64, True], [[np.float32, (5, 32, 64)], 2, 64, 64, False],
            [[np.float32, (26, 2560, 512)], 2, 512, 256, False], [[np.float32, (10, 33, 128)], 2, 128, 64, False],
            [[np.float16, (32, 64)], 2, 64, 32, True], [[np.float16, (32, 64)], 2, 64, 32, False],
            [[np.float32, (32, 64)], 2, 64, 64, True], [[np.float32, (32, 64)], 2, 64, 64, False],
            [[np.float32, (2560, 512)], 2, 512, 256, False], [[np.float32, (33, 128)], 2, 128, 64, False],
        ]

        for item in shape_format:
            cpu_lstm = torch.nn.LSTM(input_size=item[2], hidden_size=item[3], batch_first=True,
                                     num_layers=item[1], bidirectional=True, bias=False)
            cpu_lstm.training = item[4]
            npu_lstm = copy.deepcopy(cpu_lstm).npu()

            input1 = np.random.uniform(0, 1, item[0][1]).astype(np.float16).astype(np.float32)
            cpu_output_y, (cpu_output_h, cpu_output_c) = cpu_lstm(torch.from_numpy(input1))
            npu_output_y, (npu_output_h, npu_output_c) = npu_lstm(torch.from_numpy(input1.astype(item[0][0])).npu())

            self.assertRtolEqual(cpu_output_y.detach().numpy(),
                                 npu_output_y.cpu().to(torch.float).detach().numpy(), prec=1.e-3)
            self.assertRtolEqual(cpu_output_h.detach().numpy(),
                                 npu_output_h.cpu().to(torch.float).detach().numpy(), prec=1.e-3)
            self.assertRtolEqual(cpu_output_c.detach().numpy(),
                                 npu_output_c.cpu().to(torch.float).detach().numpy(), prec=2.e-3)

    def test_lstm_sequence(self):
        max_len = 6
        embedding_size = 2
        hidden_size = 16
        vocab_size = 20
        input_seq = [[3, 5, 12, 7, 2, ], [4, 11, 14, ], [18, 7, 3, 8, 5, 4]]
        lengths = [5, 3, 6]

        # embedding
        embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        rnn = torch.nn.LSTM(embedding_size, hidden_size)
        rnn_npu = copy.deepcopy(rnn).npu()

        # Sorting from Large to Small
        input_seq = sorted(input_seq, key=lambda tp: len(tp), reverse=True)
        lengths = sorted(lengths, key=lambda tp: tp, reverse=True)
        '''
        outputs:
        input_seq: [[18, 7, 3, 8, 5, 4], [3, 5, 12, 7, 2], [4, 11, 14]]
        lengths : [6, 5, 3]
        '''

        def pad_seq(seq, seq_len, max_length):
            # The padding subscript is 0
            pad_token = 0
            seq += [pad_token for _ in range(max_length - seq_len)]
            return seq

        # Data after padding
        pad_seqs = []
        for i, j in zip(input_seq, lengths):
            pad_seqs.append(pad_seq(i, j, max_len))

        lengths = [6, 5, 3]
        pad_seqs = torch.tensor(pad_seqs)
        embeded = embedding(pad_seqs)
        embeded = embeded.reshape(6, 3, 2)
        embeded = embeded.to(torch.float16).to(torch.float32)

        # cacl cpu
        pack = torch.nn.utils.rnn.pack_padded_sequence(embeded, lengths, batch_first=False)
        pade_outputs, (hn, cn) = rnn(pack)
        pade_outputs, others = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs, batch_first=False)

        # cacl npu
        embeded_npu = embeded.npu()
        pack = torch.nn.utils.rnn.pack_padded_sequence(embeded_npu, lengths, batch_first=False)
        pade_outputs_npu, (hn_n, cn_n) = rnn_npu(pack)
        pade_outputs_npu, others = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs_npu, batch_first=False)

        self.assertRtolEqual(pade_outputs.detach().numpy(),
                             pade_outputs_npu.cpu().to(torch.float).detach().numpy(), prec=1.e-4)

    def test_lstm_sequence_bidirection(self):
        max_len = 6
        embedding_size = 2
        hidden_size = 16
        vocab_size = 20
        input_seq = [[3, 5, 12, 7, 2, ], [4, 11, 14, ], [18, 7, 3, 8, 5, 4]]
        lengths = [5, 3, 6]

        # embedding
        embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        rnn = torch.nn.LSTM(embedding_size, hidden_size, num_layers=1, bidirectional=True, bias=False)
        rnn_npu = copy.deepcopy(rnn).npu()

        # Sorting from Large to Small
        input_seq = sorted(input_seq, key=lambda tp: len(tp), reverse=True)
        lengths = sorted(lengths, key=lambda tp: tp, reverse=True)
        '''
        outputs:
        input_seq: [[18, 7, 3, 8, 5, 4], [3, 5, 12, 7, 2], [4, 11, 14]]
        lengths : [6, 5, 3]
        '''

        def pad_seq(seq, seq_len, max_length):
            # The padding subscript is 0
            pad_token = 0
            seq += [pad_token for _ in range(max_length - seq_len)]
            return seq

        # Data after padding
        pad_seqs = []
        for i, j in zip(input_seq, lengths):
            pad_seqs.append(pad_seq(i, j, max_len))

        lengths = [6, 5, 3]
        pad_seqs = torch.tensor(pad_seqs)
        embeded = embedding(pad_seqs)
        embeded = embeded.reshape(6, 3, 2)
        embeded = embeded.to(torch.float16).to(torch.float32)

        # cacl cpu
        pack = torch.nn.utils.rnn.pack_padded_sequence(embeded, lengths, batch_first=False)
        pade_outputs, (hn, cn) = rnn(pack)
        pade_outputs, others = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs, batch_first=False)

        # cacl npu
        embeded_npu = embeded.npu()
        pack = torch.nn.utils.rnn.pack_padded_sequence(embeded_npu, lengths, batch_first=False)
        pade_outputs_npu, (hn_n, cn_n) = rnn_npu(pack)
        pade_outputs_npu, others = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs_npu, batch_first=False)

        self.assertRtolEqual(pade_outputs.detach().numpy(),
                             pade_outputs_npu.cpu().detach().numpy(), prec=1.e-4)

    def test_lstm_sequence_double_layer(self):
        for item in [True, False]:
            max_len, embedding_size, hidden_size, vocab_size = 6, 2, 16, 20
            input_seq = [[3, 5, 12, 7, 2, ], [4, 11, 14, ], [18, 7, 3, 8, 5, 4]]
            lengths = [5, 3, 6]

            embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=0)
            rnn = torch.nn.LSTM(embedding_size, hidden_size, num_layers=2, bidirectional=item, bias=False)
            rnn_npu = copy.deepcopy(rnn).npu()

            # Sorting from Large to Small
            input_seq = sorted(input_seq, key=lambda tp: len(tp), reverse=True)
            lengths = sorted(lengths, key=lambda tp: tp, reverse=True)

            def pad_seq(seq, seq_len, max_length):
                # The padding subscript is 0
                pad_token = 0
                seq += [pad_token for _ in range(max_length - seq_len)]
                return seq

            # Data after padding
            pad_seqs = [pad_seq(i, j, max_len) for i, j in zip(input_seq, lengths)]

            lengths = [6, 5, 3]
            pad_seqs = torch.tensor(pad_seqs)
            embeded = embedding(pad_seqs)
            embeded = embeded.reshape(6, 3, 2)
            embeded = embeded.to(torch.float16).to(torch.float32)

            # cacl cpu
            pack = torch.nn.utils.rnn.pack_padded_sequence(embeded, lengths, batch_first=False)
            pade_outputs, (hn, cn) = rnn(pack)
            pade_outputs, others = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs, batch_first=False)

            # cacl npu
            pack = torch.nn.utils.rnn.pack_padded_sequence(embeded.npu(), lengths, batch_first=False)
            pade_outputs_npu, (hn_n, cn_n) = rnn_npu(pack)
            pade_outputs_npu, others = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs_npu, batch_first=False)

            self.assertRtolEqual(pade_outputs.detach().numpy(), pade_outputs_npu.cpu().detach().numpy(), prec=1.e-4)


if __name__ == "__main__":
    run_tests()
