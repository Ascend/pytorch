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
import sys
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestLayerNorm(TestCase):
    weight_grad = []

    def getWeightGrad(self, grad):
        self.weight_grad.append(grad.cpu())

    def cpu_op_exec(self, input, normalized_shape):
        input.requires_grad_(True)
        input.retain_grad()
        m = torch.nn.LayerNorm(normalized_shape=normalized_shape)
        res = m(input)
        w = torch.ones_like(res)
        res.backward(w)

        grad_output = input.grad.detach().numpy()
        grad_bias = m.bias.grad.detach().numpy()
        grad_weight = m.weight.grad.detach().numpy()
        return grad_output, grad_weight, grad_bias

    def npu_op_exec_new(self, input, normalized_shape):
        input.requires_grad_(True)
        input.retain_grad()
        input = input.npu()
        m = torch.nn.LayerNorm(normalized_shape = normalized_shape).npu() 
        m.weight.register_hook(lambda grad: self.getWeightGrad(grad))
        res = m(input)
        w = torch.ones_like(res)
        res.backward(w)

        grad_output = input.grad.cpu().detach().numpy()
        grad_bias = m.bias.grad.cpu().detach().numpy()
        grad_weight = m.weight.grad.cpu().detach().numpy()
        return grad_output, grad_weight, grad_bias

    def test_layernorm_shape_format(self, device):
        shape_format = [
                [np.float32, 3, [256, 32, 112, 112]],
                [np.float16, 3, [256, 672, 7, 7]],
                [np.float16, 3, [256, 288, 14, 14]],
                [np.float16, 3, [1024, 58, 28, 28]],
                [np.float16, 3, [1024, 116, 14, 14]],
                [np.float16, 3, [1024, 24, 112, 112]],
                [np.float16, 0, [1024, 58, 56, 56]],
                [np.float16, 0, [1024, 58, 56, 56]],
                [np.float16, 2, [1024, 24, 28, 28]],
                [np.float16, 2, [1024, 116, 28, 28]],
                [np.float16, 29, [1024, 232, 7, 7]],
                [np.float16, 29, [1024, 232, 14, 14]],
         ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 100)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)

            cpu_grad_output, cpu_grad_weight, cpu_grad_bias = self.cpu_op_exec(cpu_input, item[2][3])
            npu_grad_output, npu_grad_weight, npu_grad_bias = self.npu_op_exec_new(npu_input, item[2][3])

            cpu_grad_output = cpu_grad_output.astype(npu_grad_output.dtype)
            cpu_grad_weight = cpu_grad_weight.astype(npu_grad_weight.dtype)
            cpu_grad_bias = cpu_grad_bias.astype(npu_grad_bias.dtype)

            self.assertRtolEqual(cpu_grad_output, npu_grad_output)
            # TODO(ascend): Insufficient precision
            #npu_grad_weight精度未满足要求
            self.assertRtolEqual(cpu_grad_weight, npu_grad_weight, 1)
            self.assertRtolEqual(cpu_grad_bias, npu_grad_bias)


instantiate_device_type_tests(TestLayerNorm, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
