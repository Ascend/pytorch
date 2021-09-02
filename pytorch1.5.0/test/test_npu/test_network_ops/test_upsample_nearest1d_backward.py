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
import math
import numpy as np
import torch.nn.functional as F
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestUpsampleNearest1DBackward(TestCase):
    def cpu_op_exec(self, input, grads, size):
        input.requires_grad_(True)
        output = F.interpolate(input, size, mode="nearest")
        output.backward(grads)
        gradcpu = input.grad
        return output.detach().numpy(), gradcpu.detach().numpy()

    def cpu_op_scale_exec(self, input, grads, scale):
        input.requires_grad_(True)
        output = F.interpolate(input, scale_factor = scale, mode="nearest")
        output.backward(grads)
        gradcpu = input.grad
        return output.detach().numpy(), gradcpu.detach().numpy()

    def npu_op_exec(self, input, grads, size):
        input.requires_grad_(True)
        output = F.interpolate(input, size, mode="nearest")
        output.backward(grads)  
        gradnpu = input.grad
        gradnpu = gradnpu.to("cpu")
        output = output.to("cpu")
        return output.detach().numpy(), gradnpu.detach().numpy()

    def npu_op_scale_exec(self, input, grads, scale):
        input.requires_grad_(True)
        output = F.interpolate(input, scale_factor = scale, mode="nearest")
        output.backward(grads)  
        gradnpu = input.grad
        gradnpu = gradnpu.to("cpu")
        output = output.to("cpu")
        return output.detach().numpy(), gradnpu.detach().numpy()

    def test_upsample_nearest1d_backward_shape_format(self, device):
        test_cases = [
            [[np.float16, 0, (1, 1, 2)], [4, ]],
            [[np.float16, 0, (2, 1, 4)], [8, ]],
            [[np.float32, 3, (2, 2, 3)], [1, ]],
            [[np.float32, 0, (2, 1, 1)], [4, ]],
            [[np.float32, 0, (4, 1, 2)], [4, ]],
            [[np.float32, 0, (1, 1, 1)], [1, ]]
        ]
        for item in test_cases:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            size = list(item[0][2])
            size[2] = item[1][0]

            grad_item = []
            grad_item.append(item[0][0])
            grad_item.append(item[0][1])
            grad_item.append(size)
            cpu_grads, npu_grads = create_common_tensor(grad_item, 0, 100)

            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)

            if cpu_grads.dtype == torch.float16:
                cpu_grads = cpu_grads.to(torch.float32)

            cpu_output, cpu_grad = self.cpu_op_exec(cpu_input, cpu_grads, item[1])
            npu_output, npu_grad = self.npu_op_exec(npu_input, npu_grads, item[1])

            cpu_output = cpu_output.astype(npu_output.dtype)
            cpu_grad = cpu_grad.astype(npu_grad.dtype)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_grad, npu_grad)

    def test_upsample_nearest1d_backward_shape_format_scale(self, device):
        test_cases = [
            [[np.float16, 0, (1, 1, 2)], 2],
            [[np.float16, 0, (2, 1, 4)], 2.2],
            [[np.float32, 3, (2, 2, 3)], 0.4],
            [[np.float32, 0, (2, 1, 1)], 4],
            [[np.float32, 0, (4, 1, 2)], 2],
            [[np.float32, 0, (1, 1, 1)], 1]
        ]
        for item in test_cases:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)

            size = list(item[0][2])
            size[2] = item[1] * item[0][2][2]
            size[2] = math.floor(size[2])

            grad_item = []
            grad_item.append(item[0][0])
            grad_item.append(item[0][1])
            grad_item.append(size)
            cpu_grads, npu_grads = create_common_tensor(grad_item, 0, 100)

            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)

            if cpu_grads.dtype == torch.float16:
                cpu_grads = cpu_grads.to(torch.float32)

            cpu_output, cpu_grad = self.cpu_op_scale_exec(cpu_input, cpu_grads, item[1])
            npu_output, npu_grad = self.npu_op_scale_exec(npu_input, npu_grads, item[1])

            cpu_output = cpu_output.astype(npu_output.dtype)
            cpu_grad = cpu_grad.astype(npu_grad.dtype)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_grad, npu_grad)

instantiate_device_type_tests(TestUpsampleNearest1DBackward, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
