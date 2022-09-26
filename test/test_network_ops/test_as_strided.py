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

import sys

import torch
import numpy as np

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestAsStrided(TestCase):
    def cpu_op_exec(self, input1, size, stride, storage_offset):
        output = torch.as_strided(input1, size, stride, storage_offset)
        output = output.numpy()
        return output

    def npu_op_exec(self,input1, size, stride, storage_offset):
        output = torch.as_strided(input1, size, stride, storage_offset)
        output = output.cpu().numpy()
        return output

    def test_as_strided(self, device='npu'):
        shape_format = [
                [[np.float32, 0, [3, 3]], (2, 2), (1, 2), 0],
                [[np.float16, 0, [13, 23]], (10, 15), (1, 2), 1],
                [[np.int32, 0, [5, 5]], (3, 3), (1, 2), 1],
                [[np.float32, 2, [32, 8, 2]], (8, 6, 2), (5, 4, 1), 1],
        ]
        
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1], item[2], item[3])
            npu_output = self.npu_op_exec(npu_input1, item[1], item[2], item[3])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
