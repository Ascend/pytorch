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

import torch
import numpy as np
import sys
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestLstm(TestCase):
    def test_lstm(self, device):
        # shape_format:[[dtype, (num_step, batch_size, input_size)], input_size, hidden_size, is_training]
        shape_format = [
                        [[np.float16, (16, 32, 64)], [np.float16, (1, 32, 32)], 64, 32, True], 
                        [[np.float16, (5, 32, 64)], [np.float16, (1, 32, 32)], 64, 32, False],
                        [[np.float32, (5, 32, 64)], [np.float16, (1, 32, 64)],64, 64, True],
                        [[np.float32, (5, 32, 64)], [np.float16, (1, 32, 64)], 64, 64, False],
                        [[np.float32, (26, 2560, 512)], [np.float16, (1, 2560, 256)], 512, 256, False],
                        [[np.float32, (10, 33, 128)], [np.float32, (1, 33, 64)], 128, 64, False],
        ]

        for item in shape_format: 
            cpu_lstm = torch.nn.LSTM(input_size=item[2], hidden_size=item[3],
                     num_layers=1, bidirectional=False, bias=False)
            cpu_lstm.training = item[4]
            npu_lstm = copy.deepcopy(cpu_lstm).npu()

            cut_value = item[3]
            iw = cpu_lstm.weight_ih_l0.split(cut_value)
            hw = cpu_lstm.weight_hh_l0.split(cut_value)
            iwt = torch.cat([iw[0], iw[2], iw[1], iw[3]], 0)
            hwt = torch.cat([hw[0], hw[2], hw[1], hw[3]], 0)
            cpu_lstm.weight_ih_l0.data = iwt
            cpu_lstm.weight_hh_l0.data = hwt

            input1 = np.random.uniform(0, 1, item[0][1]).astype(np.float32)

            cpu_input1 = torch.from_numpy(input1)
            cpu_output_y, (cpu_output_h, cpu_output_c) = cpu_lstm(cpu_input1)

            npu_input1 = torch.from_numpy(input1.astype(item[0][0])).npu()
            npu_output_y, (npu_output_h, npu_output_c) = npu_lstm(npu_input1)

            self.assertRtolEqual(cpu_output_y.detach().numpy(), npu_output_y.cpu().to(torch.float).detach().numpy(), prec=1.e-1)
            self.assertRtolEqual(cpu_output_h.detach().numpy(), npu_output_h.cpu().to(torch.float).detach().numpy(), prec=1.e-1)
            self.assertRtolEqual(cpu_output_c.detach().numpy(), npu_output_c.cpu().to(torch.float).detach().numpy(), prec=1.e-1)

    def test_lstm_double_layer(self, device):
        # shape_format:[[dtype, (num_step, batch_size, input_size)], input_size, hidden_size, is_training]
        shape_format = [
                        [[np.float16, (16, 32, 64)], 64, 32, True], 
                        [[np.float16, (5, 32, 64)], 64, 32, False],
                        [[np.float32, (5, 32, 64)], 64, 64, True],
                        [[np.float32, (5, 32, 64)], 64, 64, False],
                        [[np.float32, (26, 2560, 512)], 512, 256, False],
        ]

        for item in shape_format:
            # double layer 
            lstm = torch.nn.LSTM(input_size=item[1], hidden_size=item[2], num_layers=2, bidirectional=False, bias=True)
            lstm.training = item[3]
            npu_lstm = lstm.npu()

            #h_0 and c_0 of shape (num_layers * num_directions, batch, hidden_size)
            h0 = torch.randn(2, item[0][1][1], item[2]).npu()
            c0 = torch.randn(2, item[0][1][1], item[2]).npu()

            input1 = np.random.uniform(0, 1, item[0][1]).astype(np.float32)

            npu_input1 = torch.from_numpy(input1.astype(item[0][0])).npu()
            output, (hn, cn) = npu_lstm(npu_input1,(h0, c0))

            # single layer
            lstm1 = torch.nn.LSTM(input_size=item[1], hidden_size=item[2], num_layers=1, bidirectional=False, bias=True)
            lstm1.training = item[3]
            npu_lstm1 = lstm1.npu()
            npu_lstm1.weight_ih_l0.data= npu_lstm.weight_ih_l0.data
            npu_lstm1.weight_hh_l0.data= npu_lstm.weight_hh_l0.data
            npu_lstm1.bias_hh_l0.data= npu_lstm.bias_hh_l0.data
            npu_lstm1.bias_ih_l0.data= npu_lstm.bias_ih_l0.data

            lstm2 = torch.nn.LSTM(input_size=item[2], hidden_size=item[2], num_layers=1, bidirectional=False, bias=True)
            lstm2.training = item[3]
            npu_lstm2 = lstm2.npu()
            npu_lstm2.weight_ih_l0.data= npu_lstm.weight_ih_l1.data
            npu_lstm2.weight_hh_l0.data= npu_lstm.weight_hh_l1.data
            npu_lstm2.bias_hh_l0.data= npu_lstm.bias_hh_l1.data
            npu_lstm2.bias_ih_l0.data= npu_lstm.bias_ih_l1.data

            output1, (hn1, cn1) = npu_lstm1(npu_input1, (h0[0:1,:,:], c0[0:1,:,:]))
            output2, (hn2, cn2) = npu_lstm2(output1, (h0[1:,:,:], c0[1:,:,:]))
            
            hnf = torch.cat((hn1,hn2))
            cnf = torch.cat((cn1,cn2))
            
            self.assertRtolEqual(output.cpu().detach().numpy(), output2.cpu().detach().numpy())
            self.assertRtolEqual(hn.detach().cpu().numpy(), hnf.cpu().detach().numpy())
            self.assertRtolEqual(cn.detach().cpu().numpy(), cnf.cpu().detach().numpy())

    
    #如下 测试接口 lstm.data
    def test_lstm_sequence(self, device):    
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

        iw = rnn.weight_ih_l0.split(hidden_size)
        hw = rnn.weight_hh_l0.split(hidden_size)
        ib = rnn.bias_ih_l0.split(hidden_size)
        hb = rnn.bias_hh_l0.split(hidden_size)
        iwt = torch.cat([iw[0], iw[2], iw[1], iw[3]], 0)
        hwt = torch.cat([hw[0], hw[2], hw[1], hw[3]], 0)
        ibt = torch.cat([ib[0], ib[2], ib[1], ib[3]], 0)
        hbt = torch.cat([hb[0], hb[2], hb[1], hb[3]], 0)
        rnn.weight_ih_l0.data = iwt
        rnn.weight_hh_l0.data = hwt
        rnn.bias_ih_l0.data = ibt
        rnn.bias_hh_l0.data = hbt

        #Sorting from Large to Small
        input_seq = sorted(input_seq, key = lambda tp: len(tp), reverse=True)
        lengths = sorted(lengths, key = lambda tp: tp, reverse=True)
        '''
        outputs:
        input_seq: [[18, 7, 3, 8, 5, 4], [3, 5, 12, 7, 2], [4, 11, 14]]
        lengths : [6, 5, 3]
        '''
        
        #The padding subscript is 0
        pad_token = 0
        def pad_seq(seq, seq_len, max_length):
            seq += [pad_token for _ in range(max_length - seq_len)]
            return seq
        
        #Data after padding
        pad_seqs = [] 
        for i,j in zip(input_seq, lengths):
            pad_seqs.append(pad_seq(i, j, max_len))
            
        lengths = [6,5,3]
        pad_seqs = torch.tensor(pad_seqs)
        embeded = embedding(pad_seqs)
        embeded = embeded.reshape(6,3,2)

        #cacl cpu
        pack = torch.nn.utils.rnn.pack_padded_sequence(embeded, lengths, batch_first=False)
        pade_outputs, (hn, cn) = rnn(pack)
        pade_outputs, others = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs, batch_first=False)

        #cacl npu
        embeded_npu = embeded.npu()
        pack = torch.nn.utils.rnn.pack_padded_sequence(embeded_npu, lengths, batch_first=False)
        pade_outputs_npu, (hn_n, cn_n) = rnn_npu(pack)
        pade_outputs_npu, others = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs_npu, batch_first=False)
        
        self.assertRtolEqual(pade_outputs.detach().numpy(), 
            pade_outputs_npu.cpu().to(torch.float).detach().numpy(), prec=1.e-1)

instantiate_device_type_tests(TestLstm, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests() 
