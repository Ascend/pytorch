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


# 3d need input1's dim is 5


class TestUpsamleNearest3DBackward(TestCase):
    def get_format_fp16(self):
        shape_format = [
            [[np.float16, -1, (5, 3, 2, 6, 4)], [10, 10, 10]],
            [[np.float16, -1, (2, 3, 6, 2, 4)], [10, 10, 10]],
        ]
        return shape_format

    def get_format_fp32(self):
        shape_format = [
            [[np.float32, -1, (5, 3, 2, 6, 4)], [10, 10, 10]],
            [[np.float32, -1, (2, 3, 6, 2, 4)], [10, 10, 10]],
        ]
        return shape_format

    def cpu_op_exec(self, input1, size):
        input1.requires_grad_(True)
        output = torch.nn.functional.interpolate(input1, size, mode="nearest")
        output.sum().backward()
        output = input1.grad.numpy()
        return output

    def npu_op_exec(self, input1, size):
        input1.requires_grad_(True)
        output = torch.nn.functional.interpolate(input1, size, mode="nearest")
        output.sum().backward()
        output = input1.grad.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_scale_exec(self, input1, size):
        input1.requires_grad_(True)
        output = torch.nn.functional.interpolate(input1, scale_factor=size, mode="nearest")
        output.sum().backward()
        output = input1.grad.numpy()
        return output

    def npu_op_scale_exec(self, input1, size):
        input1.requires_grad_(True)
        output = torch.nn.functional.interpolate(input1, scale_factor=size, mode="nearest")
        output.sum().backward()
        output = input1.grad.to("cpu")
        output = output.numpy()
        return output

    def test_upsample_nearest3d_backward_shape_format(self):
        shape_format = self.get_format_fp32()
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 50)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)

            size = item[1]
            cpu_output = self.cpu_op_exec(cpu_input, size)
            npu_output = self.npu_op_exec(npu_input, size)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_upsample_nearest3d_backward_shape_format_scale(self):
        shape_format = self.get_format_fp32()
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 50)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)

            size = item[1]
            cpu_output = self.cpu_op_scale_exec(cpu_input, size)
            npu_output = self.npu_op_scale_exec(npu_input, size)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_upsample_nearest3d_backward_shape_format_fp16(self):
        shape_format = self.get_format_fp16()
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 50)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)

            size = item[1]
            cpu_output = self.cpu_op_exec(cpu_input, size)
            npu_output = self.npu_op_exec(npu_input, size)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_upsample_nearest3d_backward_shape_format_scale_fp16(self):
        shape_format = self.get_format_fp16()
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 50)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)

            size = item[1]
            cpu_output = self.cpu_op_scale_exec(cpu_input, size)
            npu_output = self.npu_op_scale_exec(npu_input, size)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_upsample_nearest3d_backward(self):
        format_list = [-1, 2, 30, 32]
        dtype_list = [np.float16, np.float32]
        shape_list = [(5, 3, 2, 6, 4), (2, 3, 6, 2, 4)]
        scalar_list = [[10, 10, 10]]
        shape_format = [
            [[d, f, s], sc] for d in dtype_list for f in format_list
            for s in shape_list for sc in scalar_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 50)
            cpu_input = cpu_input.to(torch.float32)
            size = item[1]
            cpu_output = self.cpu_op_exec(cpu_input, size)
            npu_output = self.npu_op_exec(npu_input, size)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

            cpu_output = self.cpu_op_scale_exec(cpu_input, size)
            npu_output = self.npu_op_scale_exec(npu_input, size)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_upsample_nearest3d_backward_fp16(self, device="npu"):
        cpu_x = torch.randn(10, 56, 56, 96, 11).half()
        npu_x = cpu_x.npu()
        cpu_x.requires_grad = True
        npu_x.requires_grad = True
        size = (3, 4, 2)
        cpu_out = torch.nn.functional.interpolate(cpu_x.float(), size, mode="nearest").half()
        npu_out = torch.nn.functional.interpolate(npu_x, size, mode="nearest")
        cpu_out.backward(torch.ones_like(cpu_out))
        npu_out.backward(torch.ones_like(npu_out))
        self.assertRtolEqual(cpu_x.grad, npu_x.grad.cpu())
        self.assertRtolEqual(cpu_out.detach(), npu_out.cpu().detach())


if __name__ == "__main__":
    run_tests()
