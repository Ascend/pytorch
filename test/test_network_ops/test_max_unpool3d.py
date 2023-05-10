# Copyright (c) 2020, Huawei Technologies.All rights reserved.
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
import torch.nn as nn
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestMaxUnpool3d(TestCase):
    def cpu_op_exec(self, input1):
        pool = nn.MaxPool3d(3, stride=2, return_indices=True)
        unpool = nn.MaxUnpool3d(3, stride=2)
        output, indices = pool(input1)
        unpooled_output = unpool(output, indices)
        return unpooled_output

    def npu_op_exec(self, input1):
        pool = nn.MaxPool3d(3, stride=2, return_indices=True)
        unpool = nn.MaxUnpool3d(3, stride=2).npu()
        if input1.dtype == torch.float16:
            output, indices = pool(input1.cpu().float())
            output = output.half()
        else:
            output, indices = pool(input1.cpu())
        unpooled_output = unpool(output.npu(), indices.npu())
        unpooled_output = unpooled_output.cpu()
        return unpooled_output

    def test_max_unpool3d_shape_format(self, device="npu"):
        dtype_list = [np.float32, np.float16]
        format_list = [-1]
        shape_list = [(20, 16, 51, 33, 15)]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -2, 2)
            if cpu_input.dtype == torch.float16:
                cpu_output = self.cpu_op_exec(cpu_input.float()).half()
            else:
                cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
