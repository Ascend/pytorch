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

import copy
import sys
import unittest
import torch
import numpy as np
import torch.nn as nn
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestConvTranspose3dBackward(TestCase):
    weight_grad = []
    input_grad = []

    def getWeightGrad(self, grad):
        self.weight_grad.append(grad.to("cpu"))

    def getInputGrad(self, grad):
        self.input_grad.append(grad.to("cpu"))

    def cpu_op_exec(self, input1, weight, bias1):
        input1.requires_grad = True
        input1.register_hook(lambda grad: self.getInputGrad(grad))
        weight.requires_grad = True
        weight.register_hook(lambda grad: self.getWeightGrad(grad))
        bias1.requires_grad = True

        res_forward = nn.functional.conv_transpose3d(input1, weight, padding=1, bias=bias1)
        grads = torch.ones_like(res_forward).float()
        res_forward.backward(grads, retain_graph=True)
        return res_forward

    def npu_op_exec(self, input1, weight, bias1):
        input1.requires_grad = True
        input1.register_hook(lambda grad: self.getInputGrad(grad))
        weight.requires_grad = True
        weight.register_hook(lambda grad: self.getWeightGrad(grad))
        bias1 = bias1.to("npu")
        bias1.requires_grad = True

        res_forward = nn.functional.conv_transpose3d(input1, weight, padding=1, bias=bias1)
        grads = torch.ones_like(res_forward).float()
        grads = grads.to("npu")
        res_forward.backward(grads, retain_graph=True)
        res_forward = res_forward.to("cpu")
        return res_forward

    @unittest.skip("skip test_conv_transpose3d_backward_shape_format_fp16 now")
    def test_conv_transpose3d_backward_shape_format_fp16(self):
        shape_format = [
            [[np.float16, 30, [12, 12, 4, 14, 14]], [np.float16, 30, [12, 12, 3, 3, 3]], ],
            [[np.float16, 30, [12, 64, 4, 14, 14]], [np.float16, 30, [64, 64, 3, 3, 3]], ],
            [[np.float16, 30, [12, 25, 2, 7, 7]], [np.float16, 30, [25, 25, 3, 3, 3]], ],
            [[np.float16, 30, [12, 51, 1, 4, 4]], [np.float16, 30, [51, 51, 3, 3, 3]], ],
            [[np.float16, 30, [12, 50, 2, 7, 7]], [np.float16, 30, [50, 50, 3, 3, 3]], ],
        ]
        for item in shape_format:
            cpu_bias = torch.randn(item[1][2][1])
            npu_bias = copy.deepcopy(cpu_bias)
            self.weight_grad.clear()
            self.input_grad.clear()
            cpu_input1, npu_input1 = create_common_tensor(item[0], -2, 2)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -2, 2)
            if cpu_input2.dtype == torch.float16:
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, bias1=cpu_bias)
            npu_output = self.npu_op_exec(npu_input1, npu_input2, bias1=npu_bias)

            cpu_output = cpu_output.to(torch.float16)
            npu_output = npu_output.to(torch.float16)
            self.input_grad[0] = self.input_grad[0].to(torch.float16)
            self.input_grad[1] = self.input_grad[1].to(torch.float16)
            self.weight_grad[0] = self.weight_grad[0].to(self.weight_grad[1].dtype)

            self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().numpy())
            self.assertRtolEqual(self.input_grad[0].numpy(), self.input_grad[1].numpy())
            self.assertRtolEqual(self.weight_grad[0].numpy(), self.weight_grad[1].numpy())

    def test_conv_transpose3d_backward_shape_format_fp32(self):
        shape_format = [
            [[np.float32, 30, [12, 12, 4, 14, 14]], [np.float32, 30, [12, 12, 3, 3, 3]], ],
            [[np.float32, 30, [12, 64, 4, 14, 14]], [np.float32, 30, [64, 64, 3, 3, 3]], ],
            [[np.float32, 30, [12, 25, 2, 7, 7]], [np.float32, 30, [25, 25, 3, 3, 3]], ],
            [[np.float32, 30, [12, 51, 1, 4, 4]], [np.float32, 30, [51, 51, 3, 3, 3]], ],
            [[np.float32, 30, [12, 50, 2, 7, 7]], [np.float32, 30, [50, 50, 3, 3, 3]], ],
        ]
        for item in shape_format:
            cpu_bias = torch.randn(item[1][2][1])
            npu_bias = copy.deepcopy(cpu_bias)
            self.weight_grad.clear()
            self.input_grad.clear()
            cpu_input1, npu_input1 = create_common_tensor(item[0], -2, 2)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -2, 2)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, bias1=cpu_bias)
            npu_output = self.npu_op_exec(npu_input1, npu_input2, bias1=npu_bias)

            self.weight_grad[0] = self.weight_grad[0].to(self.weight_grad[1].dtype)

            self.assertRtolEqual(cpu_output.detach().numpy(), npu_output.detach().numpy(), prec=1.e-1)
            self.assertRtolEqual(self.input_grad[0].numpy(), self.input_grad[1].numpy(), prec=1.e-1)
            self.assertRtolEqual(self.weight_grad[0].numpy(), self.weight_grad[1].numpy(), prec=1.e-1)


if __name__ == "__main__":
    run_tests()
