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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


def get_weight_grad(self, grad):
    self.weight_grad.append(grad.to("cpu"))


def get_input_grad(self, grad):
    self.input_grad.append(grad.to("cpu"))


class TestSlowConvDilated2dBackward(TestCase):
    weight_grad = []
    input_grad = []

    def cpu_op_exec(self, input1, weight, bias1, stride=1, padding=0, dilation=2, groups=1):
        weight1 = weight
        input1.requires_grad = True
        weight.requires_grad = True
        bias1.requires_grad = True

        res_forward = torch.nn.functional.conv2d(input1, weight, bias1, stride, padding, dilation, groups)
        grads = torch.ones_like(res_forward).float()
        res_forward.backward(grads)
        res_forward = res_forward.detach().numpy()
        return res_forward, input1.grad, weight.grad

    def npu_op_exec(self, input1, weight, bias1, stride=1, padding=0, dilation=2, groups=1):
        weight1 = weight

        input1.requires_grad = True
        weight.requires_grad = True
        bias1 = bias1.to("npu")
        bias1.requires_grad = True

        res_forward = torch.nn.functional.conv2d(input1, weight, bias1, stride, padding, dilation, groups)

        grads = torch.ones_like(res_forward).float()
        res_forward.backward(grads)
        grads = grads.to("npu")
        res_forward = res_forward.to("cpu")
        res_forward = res_forward.detach().numpy()
        return res_forward, input1.grad.to("cpu"), weight.grad.to("cpu")

    def test_slow_conv_dilated2d_backward_shape_format_fp16(self):
        self._test_slow_conv_dilated2d_backward_shape_format(np.float16)

    def test_slow_conv_dilated2d_backward_shape_format_fp32(self):
        self._test_slow_conv_dilated2d_backward_shape_format(np.float32)

    def _test_slow_conv_dilated2d_backward_shape_format(self, dtype):
        np.random.seed(1234)
        weight_grad = []
        input_grad = []

        shape_format = [
            [dtype, 0, (64, 1, 16, 14)],
            [dtype, 3, (256, 1, 8, 8)],
            [dtype, 4, (32, 1, 8, 8)],
            [dtype, 0, (10, 1, 16, 16)],
            [dtype, 0, (64, 1, 16, 14)],
            [dtype, 3, (256, 1, 8, 8)],
            [dtype, 4, (32, 1, 8, 8)],
            [dtype, 0, (10, 1, 16, 16)]
        ]

        for item in shape_format:
            self.weight_grad.clear()
            self.input_grad.clear()
            cpu_input1, npu_input1 = create_common_tensor(item, -2, 2)
            cpu_weight, npu_weight = create_common_tensor([item[0], 0, (3, 1, 2, 2)], -2, 2)
            cpu_bias, npu_bias = create_common_tensor([item[0], 0, (3)], 1, 100)
            dtype = item[0]
            if dtype == np.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_weight = cpu_weight.to(torch.float32)
                cpu_bias = cpu_bias.to(torch.float32)
            cpu_output, cpu_input_grad, cpu_weight_grad = self.cpu_op_exec(cpu_input1, cpu_weight, bias1=cpu_bias)
            npu_output, npu_input_grad, npu_weight_grad = self.npu_op_exec(npu_input1, npu_weight, bias1=npu_bias)
            if dtype == np.float16:
                cpu_output = cpu_output.astype(np.float16)
                cpu_input_grad = cpu_input_grad.to(torch.float16)
                cpu_weight_grad = cpu_weight_grad.to(torch.float16)
            self.assertRtolEqual(cpu_output, npu_output, 0.001)
            self.assertRtolEqual(cpu_input_grad, npu_input_grad, 0.01)
            self.assertRtolEqual(cpu_weight_grad, npu_weight_grad, 0.01)


if __name__ == "__main__":
    run_tests()
