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


class TestGru(TestCase):
    def test_gru(self, device):
        # shape_format:[[dtype, (num_step, batch_size, input_size)], input_size, hidden_size]
        shape_format = [
                        [[np.float16, (16, 32, 64)], [np.float16, (1, 32, 32)], 64, 32], 
                        [[np.float16, (2, 32, 64)], [np.float16, (1, 32, 32)], 64, 32],
                        [[np.float32, (5, 32, 64)], [np.float32, (1, 32, 32)], 64, 32],
                        [[np.float32, (10, 33, 128)], [np.float32, (1, 33, 64)], 128, 64],
        ]

        for item in shape_format: 
            cpu_gru = torch.nn.GRU(input_size=item[2], hidden_size=item[3], num_layers=1, bidirectional=False)
            npu_gru = copy.deepcopy(cpu_gru).npu()
            
            input1 = np.random.uniform(0, 1, item[0][1]).astype(item[0][0])
            if item[0][0] == np.float16:
                cpu_input1 = torch.from_numpy(input1.astype(np.float32))    # cpu only support fp32
            else:
                cpu_input1 = torch.from_numpy(input1)
            npu_input1 = torch.from_numpy(input1).npu()

            h0 = np.random.uniform(0, 1, item[1][1]).astype(item[1][0])
            if item[1][0] == np.float16:
                cpu_h0 = torch.from_numpy(h0.astype(np.float32))    # cpu only support fp32
            else:
                cpu_h0 = torch.from_numpy(h0)
            npu_h0 = torch.from_numpy(h0).npu()

            cpu_output_y, cpu_output_h = cpu_gru(cpu_input1, cpu_h0)
            npu_output_y, npu_output_h = npu_gru(npu_input1, npu_h0)

            self.assertRtolEqual(cpu_output_y.detach().numpy(), npu_output_y.cpu().detach().numpy().astype(np.float32), prec=1.e-1)
            self.assertRtolEqual(cpu_output_h.detach().numpy(), npu_output_h.cpu().detach().numpy().astype(np.float32), prec=1.e-1)


instantiate_device_type_tests(TestGru, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()