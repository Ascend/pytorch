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

from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestLstmCellBackward(TestCase):
    def test_lstm_cell_backward(self, device):
        # shape_format:[[dtype, (batch_size, input_size), input_size, hidden_size]
        shape_format = [
                        [[np.float16, (32, 64)], 64, 32], 
                        [[np.float16, (114, 24)], 64, 64],
                        [[np.float16, (36, 128)], 128, 64],
                        [[np.float32, (32, 64)], 64, 32], 
                        [[np.float32, (114, 24)], 64, 64],
                        [[np.float32, (36, 128)], 128, 64],
        ]
        for item in shape_format: 
            cpu_lstm = torch.nn.LSTMCell(input_size=item[1], hidden_size=item[2])
            npu_lstm = copy.deepcopy(cpu_lstm).npu()
            input1 = np.random.uniform(0, 1, item[0][1]).astype(np.float32)
            cpu_input1 = torch.from_numpy(input1)
            cpu_input1.requires_grad_(True)
            cpu_output_h, cpu_output_c = cpu_lstm(cpu_input1)
            npu_input1 = torch.from_numpy(input1.astype(item[0][0])).npu()
            npu_input1.requires_grad_(True)
            npu_output_h, npu_output_c = npu_lstm(npu_input1)
            self.assertRtolEqual(cpu_output_h.detach().numpy(), 
              npu_output_h.cpu().to(torch.float).detach().numpy(), prec=1.e-3)
            self.assertRtolEqual(cpu_output_c.detach().numpy(), 
              npu_output_c.cpu().to(torch.float).detach().numpy(), prec=1.e-3)

            cpu_output_c.backward(torch.ones_like(cpu_output_c, dtype=torch.float))
            cpu_dx = cpu_input1.grad
            npu_output_c.backward(torch.ones_like(npu_output_c, dtype=torch.float).npu())
            npu_dx = npu_input1.grad
            self.assertRtolEqual(cpu_dx.numpy(), npu_dx.cpu().to(torch.float).numpy(), prec=1.e-3)

instantiate_device_type_tests(TestLstmCellBackward, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests() 
