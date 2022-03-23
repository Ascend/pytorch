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


class TestAdaptiveMaxPool2dBackward(TestCase):
    def cpu_op_exec(self, input_tensor, output_size):
        input_tensor.requires_grad = True
        m = nn.AdaptiveMaxPool2d(output_size)
        output = m(input_tensor)
        output.backward(output)
        cpu_grad = input_tensor.grad
        return cpu_grad

    def npu_op_exec(self, input_tensor, output_size):
        input_tensor.requires_grad = True
        m = nn.AdaptiveMaxPool2d(output_size)
        output = m(input_tensor)
        output.backward(output)
        npu_grad = input_tensor.grad
        npu_grad = npu_grad.to("cpu")
        return npu_grad

    def test_adaptive_max_pool2d_shape_format_fp32_6(self, device="npu"):
        format_list = [-1]
        shape_list = [(1, 3, 8, 9)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        output_list = [(2, 3)]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            for output_size in output_list:
                cpu_input = cpu_input.to(torch.float32)
                cpu_output = self.cpu_op_exec(cpu_input, output_size)
                cpu_output = cpu_output.to(torch.float16)
                npu_output = self.npu_op_exec(npu_input, output_size)
                self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
