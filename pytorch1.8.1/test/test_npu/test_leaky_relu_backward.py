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

def input_grad_hook(grad):
    global input_grad
    input_grad = grad


def npu_input_grad_hook(grad):
    global npu_input_grad
    npu_input_grad = grad.to("cpu")


class TestLeakyReluBackward(TestCase):

    def cpu_op_exec(self, input, negative_slope=0):
        input.requires_grad = True
        input.register_hook(input_grad_hook)

        output = torch.nn.functional.leaky_relu(input, negative_slope=negative_slope)
        z = output.sum()
        z.backward()

    def npu_op_exec(self, input, negative_slope=0):
        input.requires_grad = True
        input.register_hook(npu_input_grad_hook)

        output = torch.nn.functional.leaky_relu(input, negative_slope=negative_slope)
        z = output.sum()
        z.backward()
        input = input.cpu()

    def test_leaky_relu_backward_shape_format_fp32(self, device):
        shape_format = [
            [[np.float32, 0, (3, 3)], 2],
            [[np.float32, 0, (64, 64)], 5],
            [[np.float32, 0, (4, 5, 6)], -3],
            [[np.float32, 0, (3, 3, 3, 4)], 0.8],
            [[np.float32, 0, (1, 2, 3, 4, 5)], -0.9]
        ]
        for item in shape_format:
            input, npu_input = create_common_tensor(item[0], 1, 100)

            self.cpu_op_exec(input, item[1])
            self.npu_op_exec(npu_input, item[1])
            self.assertRtolEqual(input_grad.numpy(), npu_input_grad.numpy())

    def test_leaky_relu_backward_shape_format_fp16(self, device):
        shape_format = [
            [[np.float16, 0, (3, 3)], 2],
            [[np.float16, 0, (64, 64)], 5],
            [[np.float16, 0, (4, 5, 6)], -3],
            [[np.float16, 0, (3, 3, 3, 4)], 0.8],
            [[np.float16, 0, (1, 2, 3, 4, 5)], -0.9]
        ]
        for item in shape_format:
            input, npu_input = create_common_tensor(item[0], 1, 100)

            input = input.to(torch.float32)
            self.cpu_op_exec(input, item[1])
            self.npu_op_exec(npu_input, item[1])
            self.assertRtolEqual(input_grad.numpy().astype(np.float16), npu_input_grad.numpy().astype(np.float16))


instantiate_device_type_tests(TestLeakyReluBackward, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
