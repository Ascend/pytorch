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

option = {}
option['NPU_FUZZY_COMPILE_BLACKLIST'] = "DynamicRNNV2"
torch.npu.set_option(option)


class TestLstmCellBackward(TestCase):
    # shape_format:[[dtype, (batch_size, input_size), input_size, hidden_size]
    shape_format = [
        [[np.float16, (32, 64)], 64, 32],
        [[np.float16, (114, 34)], 34, 64],
        [[np.float16, (36, 128)], 128, 17],
    ]

    def lstm_cell_backward_result(self, item):
        cpu_lstm = torch.nn.LSTMCell(input_size=item[1], hidden_size=item[2])
        npu_lstm = copy.deepcopy(cpu_lstm).npu()

        cpu_lstm.bias_ih.data = cpu_lstm.bias_ih.half().float()
        cpu_lstm.bias_hh.data = cpu_lstm.bias_hh.half().float()
        cpu_lstm.weight_ih.data = cpu_lstm.weight_ih.half().float()
        cpu_lstm.weight_hh.data = cpu_lstm.weight_hh.half().float()

        input1 = np.random.uniform(0, 1, item[0][1]).astype(np.float16)
        h0 = np.random.uniform(0, 1, (item[0][1][0], item[2])).astype(np.float16)
        c0 = np.random.uniform(0, 1, (item[0][1][0], item[2])).astype(np.float16)

        cpu_input1 = torch.from_numpy(input1).float()
        cpu_h0 = torch.from_numpy(h0).float()
        cpu_c0 = torch.from_numpy(c0).float()

        npu_input1 = torch.from_numpy(input1.astype(item[0][0])).npu()
        npu_h0 = torch.from_numpy(h0.astype(item[0][0])).npu()
        npu_c0 = torch.from_numpy(c0.astype(item[0][0])).npu()

        cpu_h0.requires_grad_(True)
        cpu_c0.requires_grad_(True)
        cpu_input1.requires_grad_(True)
        cpu_lstm.weight_ih.requires_grad_(True)
        cpu_lstm.weight_hh.requires_grad_(True)
        cpu_lstm.bias_ih.requires_grad_(True)
        cpu_lstm.bias_hh.requires_grad_(True)

        npu_h0.requires_grad_(True)
        npu_c0.requires_grad_(True)
        npu_input1.requires_grad_(True)

        npu_lstm.weight_ih.requires_grad_(True)
        npu_lstm.weight_hh.requires_grad_(True)
        npu_lstm.bias_ih.requires_grad_(True)
        npu_lstm.bias_hh.requires_grad_(True)

        cpu_output_h, cpu_output_c = cpu_lstm(cpu_input1, (cpu_h0, cpu_c0))
        npu_output_h, npu_output_c = npu_lstm(npu_input1, (npu_h0, npu_c0))
        input2 = torch.randn(item[0][1][0], item[2]).half().float()
        cpu_output_h.backward(input2)
        npu_output_h.backward(input2.npu())

        cpu_dh = cpu_h0.grad
        cpu_dc = cpu_c0.grad
        cpu_dx = cpu_input1.grad
        npu_dh = npu_h0.grad
        npu_dc = npu_c0.grad
        npu_dx = npu_input1.grad
        cpu_out = [cpu_dx, cpu_dh, cpu_dc]
        npu_out = [npu_dx, npu_dh, npu_dc]

        return cpu_out, npu_out, cpu_lstm, npu_lstm

    def test_lstm_cell_dx(self):
        for item in self.shape_format:
            cpu_out, npu_out, _, _ = self.lstm_cell_backward_result(item)
            cpu_dx = cpu_out[0]
            npu_dx = npu_out[0]
            self.assertRtolEqual(cpu_dx.numpy(), npu_dx.cpu().to(torch.float).numpy(), prec=1.e-3)

    def test_lstm_cell_dh(self):
        for item in self.shape_format:
            cpu_out, npu_out, _, _ = self.lstm_cell_backward_result(item)
            cpu_dh = cpu_out[1]
            npu_dh = npu_out[1]
            self.assertRtolEqual(cpu_dh.numpy(), npu_dh.cpu().to(torch.float).numpy(), prec=1.e-3)

    def test_lstm_cell_dc(self):
        for item in self.shape_format:
            cpu_out, npu_out, _, _ = self.lstm_cell_backward_result(item)
            cpu_dc = cpu_out[2]
            npu_dc = npu_out[2]
            self.assertRtolEqual(cpu_dc.numpy(), npu_dc.cpu().to(torch.float).numpy(), prec=0.002)

    def test_lstm_cell_db(self):
        for item in self.shape_format:
            _, nout, cpu_lstm, npu_lstm = self.lstm_cell_backward_result(item)
            cpu_db_ih = cpu_lstm.bias_ih.grad
            cpu_db_hh = cpu_lstm.bias_hh.grad
            npu_db_ih = npu_lstm.bias_ih.grad
            npu_db_hh = npu_lstm.bias_hh.grad
            # error in accuracy
            if nout[0].dtype == torch.float:
                continue
            self.assertRtolEqual(cpu_db_ih.numpy(), npu_db_ih.cpu().to(torch.float).numpy(), prec=0.006)
            self.assertRtolEqual(cpu_db_hh.numpy(), npu_db_hh.cpu().to(torch.float).numpy(), prec=0.006)

    def test_lstm_cell_dw(self):
        for item in self.shape_format:
            _, _, cpu_lstm, npu_lstm = self.lstm_cell_backward_result(item)
            cpu_dw_ih = cpu_lstm.weight_ih.grad
            cpu_dw_hh = cpu_lstm.weight_hh.grad
            npu_dw_ih = npu_lstm.weight_ih.grad
            npu_dw_hh = npu_lstm.weight_hh.grad
            # error in accuracy
            self.assertRtolEqual(cpu_dw_ih.numpy(), npu_dw_ih.cpu().to(torch.float).numpy(), prec=0.004)
            self.assertRtolEqual(cpu_dw_hh.numpy(), npu_dw_hh.cpu().to(torch.float).numpy(), prec=0.004)


if __name__ == "__main__":
    run_tests()
