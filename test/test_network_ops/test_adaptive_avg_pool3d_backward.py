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
import numpy as np
from torch.nn import functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestAdaptiveAvgPool3dBackward(TestCase):

    def cpu_op_exec(self, input_x, output_size):
        input_x.requires_grad_(True)
        m = torch.nn.AdaptiveAvgPool3d(output_size)
        output = m(input_x)
        ones = torch.ones_like(output)
        output.backward(ones)
        out = input_x.grad
        return out.numpy()

    def npu_op_exec(self, input_x, output_size):
        input_x.requires_grad_(True)
        m = torch.nn.AdaptiveAvgPool3d(output_size)
        output = m(input_x)
        ones = torch.ones_like(output)
        output.backward(ones)
        out = input_x.grad.cpu()
        return out.numpy()

    def test_adaptive_avg_pool3d_backward(self, device="npu"):
        dtype_list = [np.float16, np.float32]
        format_list = [-1]
        shape_list = [
            [2, 3, 7, 7],
            [1, 2, 3, 6, 6],
            [6, 5, 8, 10],
            [2, 5, 6, 8, 9]
        ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
        output_sizes = [[1, 1, 1]]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 10)
            for output_size in output_sizes:
                cpu_output = self.cpu_op_exec(cpu_input, output_size)
                npu_output = self.npu_op_exec(npu_input, output_size)
                self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
