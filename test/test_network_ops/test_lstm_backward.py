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
import torch_npu
import numpy as np
import copy

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor

class TestLstmBackward(TestCase):
    def test_lstm_backward(self, device="npu"):
        # shape_format:[[dtype, (num_step, batch_size, input_size)], input_size, hidden_size, is_training]
        shape_format = [
                        [[np.float16, (16, 32, 64)], 64, 32, True], 
                        [[np.float16, (5, 32, 64)], 64, 32, False],
                        [[np.float32, (5, 32, 64)], 64, 64, True],
                        [[np.float32, (5, 32, 64)], 64, 64, False],
                        [[np.float32, (26, 2560, 512)], 512, 256, False],
        ]

        for item in shape_format: 
            cpu_lstm = torch.nn.LSTM(input_size=item[1], hidden_size=item[2], num_layers=1, bidirectional=False, bias=False)
            cpu_lstm.training = item[3]
            npu_lstm = copy.deepcopy(cpu_lstm).npu()

            cut_value = item[2]
            iw = cpu_lstm.weight_ih_l0.split(cut_value)
            hw = cpu_lstm.weight_hh_l0.split(cut_value)
            iwt = torch.cat([iw[0], iw[2], iw[1], iw[3]], 0)
            hwt = torch.cat([hw[0], hw[2], hw[1], hw[3]], 0)
            cpu_lstm.weight_ih_l0.data = iwt
            cpu_lstm.weight_hh_l0.data = hwt

            input1 = np.random.uniform(0, 1, item[0][1]).astype(np.float32)

            cpu_input1 = torch.from_numpy(input1)
            cpu_input1.requires_grad_(True)
            cpu_output_y, (cpu_output_h, cpu_output_c) = cpu_lstm(cpu_input1)

            npu_input1 = torch.from_numpy(input1.astype(item[0][0])).npu()
            npu_input1.requires_grad_(True)
            npu_output_y, (npu_output_h, npu_output_c) = npu_lstm(npu_input1)

            self.assertRtolEqual(cpu_output_y.detach().numpy(), npu_output_y.cpu().to(torch.float).detach().numpy(), prec=1.e-3)
            self.assertRtolEqual(cpu_output_h.detach().numpy(), npu_output_h.cpu().to(torch.float).detach().numpy(), prec=1.e-3)
            self.assertRtolEqual(cpu_output_c.detach().numpy(), npu_output_c.cpu().to(torch.float).detach().numpy(), prec=1.e-3)

            cpu_input1.retain_grad()
            cpu_output_y.backward(torch.ones(cpu_output_y.size(), dtype=torch.float))
            cpu_dx = cpu_input1.grad

            npu_input1.retain_grad()
            npu_output_y.backward(torch.ones(npu_output_y.size(), dtype=torch.float).npu())
            npu_dx = npu_input1.grad

            self.assertRtolEqual(cpu_dx.numpy(), npu_dx.cpu().to(torch.float).numpy(), prec=1.e-3)
            
    def test_lstm_bidirection_backward(self, device="npu"):
        # shape_format:[[dtype, (num_step, batch_size, input_size)], input_size, hidden_size, is_training]
        shape_format = [
                        [[np.float16, (16, 32, 64)], 64, 32, True], 
                        [[np.float16, (5, 32, 64)], 64, 32, False],
                        [[np.float32, (5, 32, 64)], 64, 64, True],
                        [[np.float32, (5, 32, 64)], 64, 64, False],
                        [[np.float32, (26, 2560, 512)], 512, 256, False],
        ]

        for item in shape_format: 
            cpu_lstm = torch.nn.LSTM(input_size=item[1], hidden_size=item[2], num_layers=1, bidirectional=True, bias=False)
            cpu_lstm.training = item[3]
            npu_lstm = copy.deepcopy(cpu_lstm).npu()
            
            cut_value = item[2]
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

            input1 = np.random.uniform(0, 1, item[0][1]).astype(np.float32)

            cpu_input1 = torch.from_numpy(input1)
            cpu_input1.requires_grad_(True)
            cpu_output_y, (cpu_output_h, cpu_output_c) = cpu_lstm(cpu_input1)

            npu_input1 = torch.from_numpy(input1.astype(item[0][0])).npu()
            npu_input1.requires_grad_(True)
            npu_output_y, (npu_output_h, npu_output_c) = npu_lstm(npu_input1)

            self.assertRtolEqual(cpu_output_y.detach().numpy(), npu_output_y.cpu().to(torch.float).detach().numpy(), prec=1.e-3)
            self.assertRtolEqual(cpu_output_h.detach().numpy(), npu_output_h.cpu().to(torch.float).detach().numpy(), prec=1.e-3)
            self.assertRtolEqual(cpu_output_c.detach().numpy(), npu_output_c.cpu().to(torch.float).detach().numpy(), prec=1.e-3)

            cpu_input1.retain_grad()
            cpu_output_y.backward(torch.ones(cpu_output_y.size(), dtype=torch.float))
            cpu_dx = cpu_input1.grad

            npu_input1.retain_grad()
            npu_output_y.backward(torch.ones(npu_output_y.size(), dtype=torch.float).npu())
            npu_dx = npu_input1.grad

            self.assertRtolEqual(cpu_dx.numpy(), npu_dx.cpu().to(torch.float).numpy(), prec=1.e-3)

if __name__ == "__main__":
    run_tests() 
