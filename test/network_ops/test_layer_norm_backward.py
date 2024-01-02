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


class TestLayerNorm(TestCase):
    weight_grad = []

    def getWeightGrad(self, grad):
        self.weight_grad.append(grad.cpu())

    def cpu_op_exec(self, input1, normalized_shape):
        input1.requires_grad_(True)
        input1.retain_grad()
        m = torch.nn.LayerNorm(normalized_shape=normalized_shape)
        res = m(input1)
        w = torch.ones_like(res)
        res.backward(w)

        grad_output = input1.grad.detach().numpy()
        grad_bias = m.bias.grad.detach().numpy()
        grad_weight = m.weight.grad.detach().numpy()
        return grad_output, grad_weight, grad_bias

    def npu_op_exec_new(self, input1, normalized_shape):
        input1.requires_grad_(True)
        input1.retain_grad()
        input1 = input1.npu()
        m = torch.nn.LayerNorm(normalized_shape=normalized_shape).npu()
        m.weight.register_hook(lambda grad: self.getWeightGrad(grad))
        res = m(input1)
        w = torch.ones_like(res)
        res.backward(w)

        grad_output = input1.grad.cpu().detach().numpy()
        grad_bias = m.bias.grad.cpu().detach().numpy()
        grad_weight = m.weight.grad.cpu().detach().numpy()
        return grad_output, grad_weight, grad_bias

    def test_layernorm_shape_format(self, device="npu"):
        shape_format = [
            [np.float32, 3, [1, 32, 11, 112]],
            [np.float16, 3, [1, 67, 7, 7]],
            [np.float16, 3, [1, 88, 14, 14]],
            [np.float16, 3, [1, 58, 28, 28]],
            [np.float16, 3, [1, 116, 14, 14]],
            [np.float16, 3, [1, 24, 11, 112]],
            [np.float16, 0, [1, 8, 56, 56]],
            [np.float16, 0, [1, 8, 56, 56]],
            [np.float16, 2, [1, 24, 28, 28]],
            [np.float16, 2, [1, 16, 28, 28]],
            [np.float16, 29, [1, 232, 7, 7]],
            [np.float16, 29, [1, 23, 14, 14]],
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
            # npu_grad_weight精度未满足要求
            self.assertRtolEqual(cpu_grad_weight, npu_grad_weight)
            self.assertRtolEqual(cpu_grad_bias, npu_grad_bias)


if __name__ == "__main__":
    np.random.seed(20)
    run_tests()
