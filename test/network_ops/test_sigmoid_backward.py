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

input_grad = None
npu_input_grad = None


def input_grad_hook(grad):
    global input_grad
    input_grad = grad
    input_grad = input_grad.numpy()


def npu_input_grad_hook(grad):
    global npu_input_grad
    npu_input_grad = grad.to("cpu")
    npu_input_grad = npu_input_grad.numpy()


class TestSigmoidBackward(TestCase):
    def cpu_op_exec(self, input1, is_contiguous=True):
        if is_contiguous is False:
            input1 = input1.as_strided([2, 2], [1, 2], 1)
        input1.requires_grad = True
        input1.register_hook(input_grad_hook)
        output = torch.sigmoid(input1)
        z = output.sum()
        z.backward()

    def npu_op_exec(self, input1, is_contiguous=True):
        if is_contiguous is False:
            input1 = input1.as_strided([2, 2], [1, 2], 1)
        input1.requires_grad = True
        input1.register_hook(npu_input_grad_hook)

        output = torch.sigmoid(input1)
        z = output.sum()
        z.backward()
        input1 = input1.cpu()

    def test_sigmoid_backward_shape_format_fp16(self, device="npu"):
        format_list = [0]
        shape_list = [5, (64, 10), (32, 3, 3), (256, 2048, 7, 7)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            input1, npu_input1 = create_common_tensor(item, 1, 100)
            input2, npu_input2 = create_common_tensor(item, 1, 100)
            input1 = input1.to(torch.float32)
            input2 = input2.to(torch.float32)
            self.cpu_op_exec(input1)
            self.npu_op_exec(npu_input1)
            global input_grad
            input_grad = input_grad.astype(npu_input_grad.dtype)
            self.assertRtolEqual(input_grad, npu_input_grad)

            self.cpu_op_exec(input2, False)
            self.npu_op_exec(npu_input2, False)
            input_grad = input_grad.astype(np.float16)
            self.assertRtolEqual(input_grad, npu_input_grad)

    def test_sigmoid_backward_shape_format_fp32(self, device="npu"):
        format_list = [0, 3, 4, 29]
        shape_list = [(256, 2048, 7, 7)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            input1, npu_input1 = create_common_tensor(item, 1, 100)
            input2, npu_input2 = create_common_tensor(item, 1, 100)
            self.cpu_op_exec(input1)
            self.npu_op_exec(npu_input1)
            self.assertRtolEqual(input_grad, npu_input_grad)

            self.cpu_op_exec(input2, False)
            self.npu_op_exec(npu_input2, False)
            self.assertRtolEqual(input_grad, npu_input_grad)


if __name__ == "__main__":
    run_tests()
