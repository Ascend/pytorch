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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor

class TestRandperm(TestCase):
    def cpu_op_exec(self, input1, dtype):
        output = torch.randperm(input1, dtype=dtype, device='cpu')
        output = output.sum()
        return output.numpy()

    def npu_op_exec(self, input1, dtype):
        output = torch.randperm(input1, dtype=dtype, device='npu')
        output = output.sum()
        output = output.cpu()
        return output.numpy()

    def test_randperm_shape_format(self):
        for n in (10, 25, 123):
            for dtype in (torch.long, torch.float32, torch.float16):
                cpu_output = self.cpu_op_exec(n, dtype)
                npu_output = self.npu_op_exec(n, dtype)
                cpu_output = cpu_output.astype(npu_output.dtype)
                self.assertRtolEqual(cpu_output, npu_output)

if __name__ == "__main__":
    run_tests()