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
from torch_npu.testing.decorator import graph_mode


class TestAvgPool2dBackward(TestCase):
    def cpu_op_exec(self, input1):
        m = nn.AvgPool2d(kernel_size=2, stride=2)
        input1.requires_grad = True
        output = m(input1)
        output.backward(torch.ones_like(output))
        output_grad = input1.grad
        output_grad = output_grad.detach().numpy()
        output = output.detach().numpy()

        return output_grad, output

    def npu_op_exec(self, input1):
        m = nn.AvgPool2d(kernel_size=2, stride=2).npu()
        input1.requires_grad = True
        output = m(input1)
        output.backward(torch.ones_like(output))
        output_grad = input1.grad
        output_grad = output_grad.to("cpu")
        output_grad = output_grad.detach().numpy()
        output = output.to("cpu")
        output = output.detach().numpy()

        return output_grad, output

    @graph_mode
    def test_avg_pool2d_backward_shape_format_fp16(self):
        format_list = [0, 3]
        shape_list = [(5, 20, 8, 8)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_input = cpu_input.to(torch.float32)
            cpu_output_grad, cpu_output = self.cpu_op_exec(cpu_input)
            npu_output_grad, npu_output = self.npu_op_exec(npu_input)
            cpu_output = cpu_output.astype(npu_output.dtype)
            cpu_output_grad = cpu_output_grad.astype(npu_output_grad.dtype)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output_grad, npu_output_grad)

    @graph_mode
    def test_avg_pool2d_backward_shape_format_fp32(self):
        format_list = [0, 3]
        shape_list = [(5, 20, 8, 8)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_output_grad, cpu_output = self.cpu_op_exec(cpu_input)
            npu_output_grad, npu_output = self.npu_op_exec(npu_input)

            cpu_output = cpu_output.astype(np.float16)
            cpu_output_grad = cpu_output_grad.astype(np.float16)
            npu_output = npu_output.astype(np.float16)
            npu_output_grad = npu_output_grad.astype(np.float16)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output_grad, npu_output_grad)


if __name__ == "__main__":
    run_tests()
