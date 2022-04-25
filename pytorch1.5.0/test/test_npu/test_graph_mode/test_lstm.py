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
from graph_utils import graph_mode

class TestLstm(TestCase):
    @graph_mode
    def test_lstm_single_direction(self, device):
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
        ]

        for item in shape_format: 
            print(item)
            cpu_lstm = torch.nn.LSTM(input_size=item[2], hidden_size=item[3], batch_first=item[5],
                     num_layers=item[1], bidirectional=False, bias=False)
            cpu_lstm.training = item[4]
            npu_lstm = copy.deepcopy(cpu_lstm).npu()

            cut_value = item[3]
            iw = cpu_lstm.weight_ih_l0.split(cut_value)
            hw = cpu_lstm.weight_hh_l0.split(cut_value)
            iwt = torch.cat([iw[0], iw[2], iw[1], iw[3]], 0)
            hwt = torch.cat([hw[0], hw[2], hw[1], hw[3]], 0)
            cpu_lstm.weight_ih_l0.data = iwt
            cpu_lstm.weight_hh_l0.data = hwt
            
            if item[1] == 2:
                iw1 = cpu_lstm.weight_ih_l1.split(cut_value)
                hw1 = cpu_lstm.weight_hh_l1.split(cut_value)
                iwt1 = torch.cat([iw1[0], iw1[2], iw1[1], iw1[3]], 0)
                hwt1 = torch.cat([hw1[0], hw1[2], hw1[1], hw1[3]], 0)
                cpu_lstm.weight_ih_l1.data = iwt1
                cpu_lstm.weight_hh_l1.data = hwt1

            input1 = np.random.uniform(0, 1, item[0][1]).astype(np.float32)

            cpu_input1 = torch.from_numpy(input1)
            cpu_output_y, (cpu_output_h, cpu_output_c) = cpu_lstm(cpu_input1)

            npu_input1 = torch.from_numpy(input1.astype(item[0][0])).npu()
            npu_output_y, (npu_output_h, npu_output_c) = npu_lstm(npu_input1)

            self.assertRtolEqual(cpu_output_y.detach().numpy(), 
              npu_output_y.cpu().to(torch.float).detach().numpy(), prec=1.e-3)
            self.assertRtolEqual(cpu_output_h.detach().numpy(), 
              npu_output_h.cpu().to(torch.float).detach().numpy(), prec=1.e-3)
            self.assertRtolEqual(cpu_output_c.detach().numpy(), 
              npu_output_c.cpu().to(torch.float).detach().numpy(), prec=1.e-3)

    @graph_mode
    def test_lstm_bidirection(self, device):
        # shape_format:[[dtype, (num_step, batch_size, input_size)], 
        # num_layers, input_size, hidden_size, is_training]
        shape_format = [
                        [[np.float16, (16, 32, 64)], 1, 64, 32, True], 
                        [[np.float16, (5, 32, 64)], 1, 64, 32, False],
                        [[np.float32, (5, 32, 64)], 1,64, 64, True],
                        [[np.float32, (5, 32, 64)], 1, 64, 64, False],
                        [[np.float32, (26, 2560, 512)], 1, 512, 256, False],
                        [[np.float32, (10, 33, 128)], 1, 128, 64, False],
                        [[np.float16, (16, 32, 64)], 2, 64, 32, True], 
                        [[np.float16, (5, 32, 64)], 2, 64, 32, False],
                        [[np.float32, (5, 32, 64)], 2,64, 64, True],
                        [[np.float32, (5, 32, 64)], 2, 64, 64, False],
                        [[np.float32, (26, 2560, 512)], 2, 512, 256, False],
                        [[np.float32, (10, 33, 128)], 2, 128, 64, False],
        ]

        for item in shape_format:
            print(item) 
            cpu_lstm = torch.nn.LSTM(input_size=item[2], hidden_size=item[3], batch_first=True,
                     num_layers=item[1], bidirectional=True, bias=False)
            cpu_lstm.training = item[4]
            npu_lstm = copy.deepcopy(cpu_lstm).npu()

            cut_value = item[3]
            iw = cpu_lstm.weight_ih_l0.split(cut_value)
            hw = cpu_lstm.weight_hh_l0.split(cut_value)
            iwr = cpu_lstm.weight_ih_l0_reverse.split(cut_value)
            hwr = cpu_lstm.weight_hh_l0_reverse.split(cut_value)
            iwt = torch.cat([iw[0], iw[2], iw[1], iw[3]], 0)
            hwt = torch.cat([hw[0], hw[2], hw[1], hw[3]], 0)
            iwrt = torch.cat([iwr[0], iwr[2], iwr[1], iwr[3]], 0)
            hwrt = torch.cat([hwr[0], hwr[2], hwr[1], hwr[3]], 0)
            cpu_lstm.weight_ih_l0.data = iwt
            cpu_lstm.weight_hh_l0.data = hwt
            cpu_lstm.weight_ih_l0_reverse.data = iwrt
            cpu_lstm.weight_hh_l0_reverse.data = hwrt
            
            if item[1] == 2:
                iw1 = cpu_lstm.weight_ih_l1.split(cut_value)
                hw1 = cpu_lstm.weight_hh_l1.split(cut_value)
                iwr1 = cpu_lstm.weight_ih_l1_reverse.split(cut_value)
                hwr1 = cpu_lstm.weight_hh_l1_reverse.split(cut_value)
                iwt1 = torch.cat([iw1[0], iw1[2], iw1[1], iw1[3]], 0)
                hwt1 = torch.cat([hw1[0], hw1[2], hw1[1], hw1[3]], 0)
                iwrt1 = torch.cat([iwr1[0], iwr1[2], iwr1[1], iwr1[3]], 0)
                hwrt1 = torch.cat([hwr1[0], hwr1[2], hwr1[1], hwr1[3]], 0)
                cpu_lstm.weight_ih_l1.data = iwt1
                cpu_lstm.weight_hh_l1.data = hwt1
                cpu_lstm.weight_ih_l1_reverse.data = iwrt1
                cpu_lstm.weight_hh_l1_reverse.data = hwrt1              

            input1 = np.random.uniform(0, 1, item[0][1]).astype(np.float32)

            cpu_input1 = torch.from_numpy(input1)
            cpu_output_y, (cpu_output_h, cpu_output_c) = cpu_lstm(cpu_input1)

            npu_input1 = torch.from_numpy(input1.astype(item[0][0])).npu()
            npu_output_y, (npu_output_h, npu_output_c) = npu_lstm(npu_input1)

            self.assertRtolEqual(cpu_output_y.detach().numpy(), 
              npu_output_y.cpu().to(torch.float).detach().numpy(), prec=1.e-3)
            self.assertRtolEqual(cpu_output_h.detach().numpy(), 
              npu_output_h.cpu().to(torch.float).detach().numpy(), prec=1.e-3)
            self.assertRtolEqual(cpu_output_c.detach().numpy(), 
              npu_output_c.cpu().to(torch.float).detach().numpy(), prec=1.e-3)

    @graph_mode
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
            pade_outputs_npu.cpu().to(torch.float).detach().numpy(), prec=1.e-4)

    @graph_mode
    def test_lstm_sequence_bidirection(self, device):    
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

        iw = rnn.weight_ih_l0.split(hidden_size)
        hw = rnn.weight_hh_l0.split(hidden_size)
        iwr = rnn.weight_ih_l0_reverse.split(hidden_size)
        hwr = rnn.weight_hh_l0_reverse.split(hidden_size)
        iwt = torch.cat([iw[0], iw[2], iw[1], iw[3]], 0)
        hwt = torch.cat([hw[0], hw[2], hw[1], hw[3]], 0)
        iwrt = torch.cat([iwr[0], iwr[2], iwr[1], iwr[3]], 0)
        hwrt = torch.cat([hwr[0], hwr[2], hwr[1], hwr[3]], 0)
        rnn.weight_ih_l0.data = iwt
        rnn.weight_hh_l0.data = hwt
        rnn.weight_ih_l0_reverse.data = iwrt
        rnn.weight_hh_l0_reverse.data = hwrt

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
            pade_outputs_npu.cpu().detach().numpy(), prec=1.e-4)

    @graph_mode
    def test_lstm_sequence_double_layer(self, device): 
        direction = [True, False]
       
        for item in direction:
            max_len = 6        
            embedding_size = 2 
            hidden_size = 16   
            vocab_size = 20
            input_seq = [[3, 5, 12, 7, 2, ], [4, 11, 14, ], [18, 7, 3, 8, 5, 4]]
            lengths = [5, 3, 6]
            
            embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=0)
            rnn = torch.nn.LSTM(embedding_size, hidden_size, num_layers=2, bidirectional=item, bias=False)
            rnn_npu = copy.deepcopy(rnn).npu()
      
            iw0 = rnn.weight_ih_l0.split(hidden_size)
            hw0 = rnn.weight_hh_l0.split(hidden_size)
            iw1 = rnn.weight_ih_l1.split(hidden_size)
            hw1 = rnn.weight_hh_l1.split(hidden_size)
            iwt0 = torch.cat([iw0[0], iw0[2], iw0[1], iw0[3]], 0)
            hwt0 = torch.cat([hw0[0], hw0[2], hw0[1], hw0[3]], 0)
            iwt1 = torch.cat([iw1[0], iw1[2], iw1[1], iw1[3]], 0)
            hwt1 = torch.cat([hw1[0], hw1[2], hw1[1], hw1[3]], 0)
            rnn.weight_ih_l0.data = iwt0
            rnn.weight_hh_l0.data = hwt0
            rnn.weight_ih_l1.data = iwt1
            rnn.weight_hh_l1.data = hwt1
            
            if item == True:
                iwr0 = rnn.weight_ih_l0_reverse.split(hidden_size)
                hwr0 = rnn.weight_hh_l0_reverse.split(hidden_size)
                iwr1 = rnn.weight_ih_l1_reverse.split(hidden_size)
                hwr1 = rnn.weight_hh_l1_reverse.split(hidden_size)
                iwrt0 = torch.cat([iwr0[0], iwr0[2], iwr0[1], iwr0[3]], 0)
                hwrt0 = torch.cat([hwr0[0], hwr0[2], hwr0[1], hwr0[3]], 0)
                iwrt1 = torch.cat([iwr1[0], iwr1[2], iwr1[1], iwr1[3]], 0)
                hwrt1 = torch.cat([hwr1[0], hwr1[2], hwr1[1], hwr1[3]], 0)
                rnn.weight_ih_l0_reverse.data = iwrt0
                rnn.weight_hh_l0_reverse.data = hwrt0
                rnn.weight_ih_l1_reverse.data = iwrt1
                rnn.weight_hh_l1_reverse.data = hwrt1
      
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
                pade_outputs_npu.cpu().detach().numpy(), prec=1.e-4)
            
instantiate_device_type_tests(TestLstm, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests() 
