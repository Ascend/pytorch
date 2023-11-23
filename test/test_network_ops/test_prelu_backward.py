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

import unittest
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestPreluBackward(TestCase):
    def cpu_op_back_exec_ext(self, input1, weight):
        w = torch.ones_like(input1)
        input1.requires_grad_(True)
        m = torch.nn.PReLU(weight)
        tmp = m(input1)
        tmp.backward(w)
        output = input1.grad
        output = output.numpy()
        return output

    def npu_op_back_exec_ext(self, input1, weight):
        w = torch.ones_like(input1)
        w = w.to("npu")
        m = torch.nn.PReLU(weight)
        m = m.to("npu")
        input1.requires_grad_(True)
        input1 = input1.to("npu")
        tmp = m(input1)
        tmp.backward(w)
        output = input1.grad.to("cpu")
        output = output.numpy()
        return output

    @unittest.skip("skip test_PreluBackward_shape_format_fp32 now")
    def test_PreluBackward_shape_format_fp32(self, device="npu"):
        shape_format = [
            [np.float32, 0, (17, 12, 38, 15)],
            [np.float32, 0, (1, 12, 38, 5)],
            [np.float32, 0, (124, 12, 38, 25)],
            [np.float32, 0, (4, 12, 38, 5)],
            [np.float32, 0, (10, 12, 38, 45)],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -2, 2)
            cpu_weight = npu_weight = torch.randn(12)
            cpu_output = self.cpu_op_back_exec_ext(cpu_input, cpu_weight)
            npu_output = self.npu_op_back_exec_ext(npu_input, npu_weight)
            self.assertRtolEqual(cpu_output, npu_output)

    @unittest.skip("skip test_PreluBackward_shape_format_fp16 now")
    def test_PreluBackward_shape_format_fp16(self, device="npu"):
        def cpu_op_back_exec_fp16_ext(input1, weight):
            input1 = input1.to(torch.float32)
            weight = weight.to(torch.float32)
            w = torch.ones_like(input1)
            input1.requires_grad_(True)
            m = torch.nn.PReLU(weight)
            tmp = m(input1)
            tmp.backward(w)
            output = input1.grad
            output = output.detach().numpy()
            output = output.astype(np.float16)
            return output
        shape_format = [
            [np.float16, 0, (3, 5, 4)],
            [np.float16, 0, (32, 1, 1)],
            [np.float16, 0, (3, 224, 224)],
            [np.float16, 0, (5, 32, 112)],
            [np.float16, 0, (2, 672, 7)],
            [np.float16, 0, (6, 288, 14)],
            [np.float16, 0, (4, 58, 28)],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -2, 2)
            cpu_weight = npu_weight = torch.randn(1)
            cpu_output = cpu_op_back_exec_fp16_ext(cpu_input, cpu_weight)
            npu_output = self.npu_op_back_exec_ext(npu_input, npu_weight)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
