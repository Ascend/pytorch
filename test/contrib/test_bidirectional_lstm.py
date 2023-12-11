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
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.contrib.module import BiLSTM


class TestBidirectionalLstm(TestCase):

    def npu_bidirectional_lstm(self, input1):
        input1 = input1.npu()
        input1.requires_grad = True
        rnn = BiLSTM(8, 4).npu()
        input1.retain_grad()
        output = rnn(input1)
        output.backward(torch.ones(input1.size(), dtype=torch.float).npu())
        input_grad = input1.grad.cpu()
        return output.detach().cpu(), input_grad.cpu()

    @unittest.skip("skip test_bidirectional_lstm now")
    def test_bidirectional_lstm(self):
        cpu_input = torch.rand(2, 2, 8)
        npu_input = cpu_input.npu()

        npu_output, npu_inputgrad = self.npu_bidirectional_lstm(npu_input)
        expedt_cpu_output = torch.tensor([[[-0.1025, -0.1874, 0.0458, -0.1486, -0.0266, 0.1953, -0.1688, 0.0765],
                                         [-0.1941, -0.2162, -0.2046, -0.1855, -0.0262, 0.1460, -0.1729, 0.1274]],
            [[-0.2140, -0.2439, -0.0682, -0.1685, -0.0381, 0.1166, -0.1262, 0.1035],
             [-0.2947, -0.2786, -0.2559, -0.1584, -0.0176, 0.1005, -0.1135, 0.1113]]],
            dtype=torch.float32)
        expedt_cpu_inputgrad = torch.tensor([[[-0.1387, 0.4024, -0.2715, -0.0965, -0.4193, 0.3688, 0.1259, -0.3764],
                                            [-0.1760, 0.4257, -0.2285, -0.2325, -0.3738, 0.3502, -0.0338, -0.3057]],
            [[-0.0190, 0.4121, -0.2733, -0.1313, -0.1804, 0.3720, -0.0196, -0.1863],
             [-0.0638, 0.4532, -0.2258, -0.2342, -0.1488, 0.3592, -0.0708, -0.2137]]],
            dtype=torch.float32)

        self.assertRtolEqual(expedt_cpu_output, npu_output)
        self.assertRtolEqual(expedt_cpu_inputgrad, npu_inputgrad)


if __name__ == "__main__":
    run_tests()
