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
import torch.nn.functional as F
import numpy as np

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor

class TestMish(TestCase):
    def npu_op_exec(self, input1):
        output = torch_npu.npu_mish(input1)
        output = output.cpu().numpy()
        return output

    def cpu_op_exec(self, input1):
        output = input1 * (torch.tanh(F.softplus(input1)))
        output = output.numpy()
        return output

    def test_mish_fp32(self, device="npu"):
        shape_format = [
            [[np.float32, -1, [10,30,10]]],
            [[np.float32, -1, [20,30,20]]],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_mish_fp16(self, device="npu"):
        shape_format = [
            [[np.float16, -1, [10,30,10]]],
            [[np.float16, -1, [20,30,20]]],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input.float()).astype(np.float16)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

if __name__ == "__main__":
    run_tests()