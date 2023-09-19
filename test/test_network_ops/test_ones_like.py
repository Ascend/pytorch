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
from torch_npu.testing.decorator import graph_mode
from torch_npu.testing.common_utils import create_common_tensor

class TestOnesLike(TestCase):
    
    def cpu_op_exec(self, input1):
        output = torch.ones_like(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.ones_like(input1)
        output = output.to('cpu')
        output = output.numpy()
        return output
    
    @graph_mode
    def test_ones_like_shape_format(self):
        shape_format = [
            [np.float32, -1, (3, )],
            [np.float32, -1, (2, 4)],
            [np.float32, -1, (3, 6, 9)],
            [np.int8, -1, (3,)],
            [np.int8, -1, (2, 4)],
            [np.int8, -1, (3, 6, 9)],
            [np.int32, -1, (3,)],
            [np.int32, -1, (2, 4)],
            [np.int32, -1, (3, 6, 9)],
            [np.uint8, -1, (3,)],
            [np.uint8, -1, (2, 4)],
            [np.uint8, -1, (3, 6, 9)]
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 100)

            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)

            self.assertRtolEqual(cpu_output, npu_output)


    @graph_mode
    def test_ones_like_float16_shape_format(self):
        shape_format = [
            [np.float16, -1, (3, )],
            [np.float16, -1, (2, 4)],
            [np.float16, -1, (3, 6, 9)],
            [np.float16, -1, (3, 4, 5, 12)]
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 100)

            cpu_input = cpu_input.to(torch.float32)

            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)

            cpu_output = cpu_output.astype(np.float16)

            self.assertRtolEqual(cpu_output, npu_output)
    
if __name__ == "__main__":
    run_tests()