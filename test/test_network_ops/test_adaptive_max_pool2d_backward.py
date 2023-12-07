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
import torch.nn.functional as F
import numpy as np

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

    def test_adaptiveMaxPool2d_shape_format_fp32_6(self):
        format_list = [0, 3]
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

    def test_adaptiveMaxPool2d_backward_case_in_photo2cartoon(self):
        cpu_x = torch.rand(1, 256, 31, 31)
        npu_x = cpu_x.npu()
        cpu_x.requires_grad = True
        npu_x.requires_grad = True
        cpu_out = F.adaptive_max_pool2d(cpu_x, 1)
        npu_out = F.adaptive_max_pool2d(npu_x, 1)
        cpu_out.backward(torch.ones_like(cpu_out))
        npu_out.backward(torch.ones_like(npu_out))
        self.assertRtolEqual(cpu_x.grad, npu_x.grad.cpu(), 0.0003)

    def test_adaptiveMaxPool2d_backward_case_in_photo2cartoon_fp16(self):
        cpu_x = torch.rand(1, 256, 31, 31).half()
        npu_x = cpu_x.npu()
        cpu_x.requires_grad = True
        npu_x.requires_grad = True
        cpu_out = F.adaptive_max_pool2d(cpu_x.float(), 1).half()
        npu_out = F.adaptive_max_pool2d(npu_x, 1)
        cpu_out.backward(torch.ones_like(cpu_out))
        npu_out.backward(torch.ones_like(npu_out))
        self.assertRtolEqual(cpu_x.grad, npu_x.grad.cpu())


if __name__ == "__main__":
    run_tests()
