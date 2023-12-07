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


class TestUpsampleLinear1DBackward(TestCase):
    def creat_shape_format(self):
        format_list = [0]
        align_list = [True, False]
        dtype_list = [np.float16, np.float32]
        shape_list = [(17, 13, 1, 15), (38, 7, 1, 7), (61, 41, 1, 1),
                      (78, 73, 1, 1), (627, 2, 1, 3), (1008, 3, 1, 2)]
        size = [[4, ], [7, ], [8, ], [15, ], [16, ], [17, ], [32, ]]

        shape_format = [[i, j, k, h, f] for i in dtype_list
                        for j in format_list for k in shape_list for h in size for f in align_list]

        return shape_format

    def cpu_op_exec(self, input1, grads, size, align_corners):
        input1.requires_grad_(True)
        output = torch._C._nn.upsample_linear1d(input1, size, align_corners=align_corners)
        output.backward(grads)
        gradcpu = input1.grad
        return output.detach().numpy(), gradcpu.detach().numpy()

    def npu_op_exec(self, input1, grads, size, align_corners):
        input1.requires_grad_(True)
        output = torch._C._nn.upsample_linear1d(input1, size, align_corners=align_corners)
        output = output.to("npu")
        output.backward(grads)
        gradnpu = input1.grad
        gradnpu = gradnpu.to("cpu")
        output = output.to("cpu")
        return output.detach().numpy(), gradnpu.detach().numpy()

    def cpu_op_scale_exec(self, input1, grads, size, align_corners):
        input1.requires_grad_(True)
        output = torch.nn.functional.interpolate(input1, scale_factor=size, mode="linear", align_corners=align_corners)
        output.backward(grads)
        gradcpu = input1.grad
        return output.detach().numpy(), gradcpu.detach().numpy()

    def npu_op_scale_exec(self, input1, grads, size, align_corners):
        input1.requires_grad_(True)
        output = torch.nn.functional.interpolate(input1, scale_factor=size, mode="linear", align_corners=align_corners)
        output = output.to("npu")
        output.backward(grads)
        gradnpu = input1.grad
        gradnpu = gradnpu.to("cpu")
        output = output.to("cpu")
        return output.detach().numpy(), gradnpu.detach().numpy()

    def test_upsample_linear1d_backward(self):
        for item in self.creat_shape_format():
            cpu_input, npu_input = create_common_tensor(item, 0, 100)

            size = list(item[2])
            size[3] = item[3][0]

            grad_item = []
            grad_item.append(item[0])
            grad_item.append(item[1])
            grad_item.append(size)
            cpu_grads, npu_grads = create_common_tensor(grad_item, 0, 100)

            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)

            if cpu_grads.dtype == torch.float16:
                cpu_grads = cpu_grads.to(torch.float32)

            if cpu_input.dim() == 4:
                cpu_input = cpu_input.squeeze(2)

            if npu_input.dim() == 4:
                npu_input = npu_input.squeeze(2)

            if cpu_grads.dim() == 4:
                cpu_grads = cpu_grads.squeeze(2)

            if npu_grads.dim() == 4:
                npu_grads = npu_grads.squeeze(2)

            size = item[3]
            align_corners = item[4]

            cpu_output, cpu_grad = self.cpu_op_exec(cpu_input, cpu_grads, size, align_corners)
            npu_output, npu_grad = self.npu_op_exec(npu_input, npu_grads, size, align_corners)

            cpu_output = cpu_output.astype(npu_output.dtype)
            cpu_grad = cpu_grad.astype(npu_grad.dtype)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_grad, npu_grad)

    def test_upsample_linear1d_backward_scale(self):
        for item in self.creat_shape_format():
            cpu_input, npu_input = create_common_tensor(item, 0, 100)

            size = list(item[2])
            size[3] = item[3][0] * item[2][3]

            grad_item = []
            grad_item.append(item[0])
            grad_item.append(item[1])
            grad_item.append(size)
            cpu_grads, npu_grads = create_common_tensor(grad_item, 0, 100)

            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)

            if cpu_grads.dtype == torch.float16:
                cpu_grads = cpu_grads.to(torch.float32)

            if cpu_input.dim() == 4:
                cpu_input = cpu_input.squeeze(2)

            if npu_input.dim() == 4:
                npu_input = npu_input.squeeze(2)

            if cpu_grads.dim() == 4:
                cpu_grads = cpu_grads.squeeze(2)

            if npu_grads.dim() == 4:
                npu_grads = npu_grads.squeeze(2)

            size = item[3]
            align_corners = item[4]
            cpu_output, cpu_grad = self.cpu_op_scale_exec(cpu_input, cpu_grads, size, align_corners)
            npu_output, npu_grad = self.npu_op_scale_exec(npu_input, npu_grads, size, align_corners)

            cpu_output = cpu_output.astype(npu_output.dtype)
            cpu_grad = cpu_grad.astype(npu_grad.dtype)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_grad, npu_grad)


if __name__ == "__main__":
    run_tests()
