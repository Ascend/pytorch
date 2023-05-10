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


class TestMaxUnpool3dBackward(TestCase):
    def cpu_op_exec(self, input1):
        input1.requires_grad = True
        pool = nn.MaxPool3d(3, stride=2, return_indices=True)
        unpool = nn.MaxUnpool3d(3, stride=2)
        output, indices = pool(input1)
        unpooled_output = unpool(output, indices)
        output.retain_grad()
        unpooled_output.backward(torch.ones_like(unpooled_output))
        unpool_input_grad = output.grad
        unpooled_output = unpooled_output.detach()
        return unpooled_output, unpool_input_grad

    def npu_op_exec(self, input1):
        input1.requires_grad = True
        pool = nn.MaxPool3d(3, stride=2, return_indices=True)
        unpool = nn.MaxUnpool3d(3, stride=2).npu()
        if input1.dtype == torch.float16:
            output, indices = pool(input1.cpu().float())
            output = output.half()
        else:
            output, indices = pool(input1.cpu())
        npu_output = output.npu()
        npu_indices = indices.npu()
        npu_output.retain_grad()
        unpooled_output = unpool(npu_output, npu_indices)
        unpooled_output.backward(torch.ones_like(unpooled_output))
        unpool_input_grad = npu_output.grad.cpu()
        unpooled_output = unpooled_output.cpu().detach()
        return unpooled_output, unpool_input_grad

    def test_max_unpool3d_backward_shape_format(self, device="npu"):
        dtype_list = [np.float32, np.float16]
        format_list = [-1]
        shape_list = [(20, 16, 51, 33, 15)]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -2, 2)
            if cpu_input.dtype == torch.float16:
                cpu_output, cpu_unpool_input_grad = self.cpu_op_exec(cpu_input.float())
                cpu_output = cpu_output.half()
                cpu_unpool_input_grad = cpu_unpool_input_grad.half()
            else:
                cpu_output, cpu_unpool_input_grad = self.cpu_op_exec(cpu_input)
            npu_output, npu_unpool_input_grad = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_unpool_input_grad, npu_unpool_input_grad)


if __name__ == "__main__":
    run_tests()
