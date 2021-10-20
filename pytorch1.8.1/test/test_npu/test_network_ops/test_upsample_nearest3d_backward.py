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
import torch.nn.functional as F
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


# 3d need input1's dim is 5
class TestUpsamleNearest3DBackward(TestCase):
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

    def test_upsample_nearest3d_backward_shape_format(self, device):
        shape_format = [
            [[np.float32, -1, (5, 3, 2, 6, 4)], [10, 10, 10]],
            [[np.float32, -1, (2, 3, 6, 2, 4)], [10, 10, 10]],
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 50)
            if cpu_input == torch.float16:
                cpu_input = cpu_input.to(torch.float32)

            size = item[1]
            cpu_output = self.cpu_op_exec(cpu_input, size)
            npu_output = self.npu_op_exec(npu_input, size)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_upsample_nearest3d_backward_shape_format_scale(self, device):
        shape_format = [
            [[np.float32, -1, (5, 3, 2, 6, 4)], [10, 10, 10]],
            [[np.float32, -1, (2, 3, 6, 2, 4)], [10, 10, 10]],
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 50)
            if cpu_input == torch.float16:
                cpu_input = cpu_input.to(torch.float32)

            size = item[1]
            cpu_output = self.cpu_op_scale_exec(cpu_input, size)
            npu_output = self.npu_op_scale_exec(npu_input, size)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestUpsamleNearest3DBackward, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
