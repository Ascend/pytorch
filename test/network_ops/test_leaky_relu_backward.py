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


class TestLeakyReluBackward(TestCase):
    def cpu_op_backward_exec(self, input1):
        w = torch.ones_like(input1)
        input1.requires_grad_(True)
        output = torch.nn.functional.leaky_relu(input1)
        output.backward(w)
        res = input1.grad
        res = res.numpy()
        return res, output

    def npu_op_backward_exec(self, input1):
        w = torch.ones_like(input1)
        w = w.to("npu")
        input1 = input1.to("npu")
        input1.requires_grad_(True)
        output = torch.nn.functional.leaky_relu(input1)
        output.backward(w)
        output = output.to("cpu")
        res = input1.grad
        res = input1.grad.to("cpu")
        res = res.numpy()
        return res, output

    def test_leaky_relu_backward_format_fp32(self, device="npu"):
        format_list = [0, 3]
        shape_list = [(5, 3)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 2)
            cpu_output = self.cpu_op_backward_exec(cpu_input1)
            npu_output = self.npu_op_backward_exec(npu_input1)
            self.assertEqual(cpu_output, npu_output)

    def test_leaky_relu_backward_format_fp16(self, device="npu"):
        format_list = [0, 3]
        shape_list = [(5, 3)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 2)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output1, cpu_output2 = self.cpu_op_backward_exec(cpu_input1)
            npu_output1, npu_output2 = self.npu_op_backward_exec(npu_input1)
            cpu_output1 = cpu_output1.astype(np.float16)
            self.assertEqual(cpu_output1, npu_output1)
            self.assertEqual(cpu_output2, npu_output2)


if __name__ == "__main__":
    run_tests()
