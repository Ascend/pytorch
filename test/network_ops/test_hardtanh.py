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
import torch.nn as nn
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestHardtanh(TestCase):
    def cpu_op_backward_exec(self, input1, min1, max1):
        w = torch.ones_like(input1)
        input1.requires_grad_(True)
        output = torch.nn.functional.hardtanh(input1, min1, max1)
        output.backward(w)
        output = output.detach().numpy()
        res = input1.grad
        res = res.numpy()
        return output, res

    def npu_op_backward_exec(self, input1, min1, max1):
        w = torch.ones_like(input1)
        w = w.to("npu")
        input1.requires_grad_(True)
        output = torch.nn.functional.hardtanh(input1, min1, max1)
        output.backward(w)
        output = output.to("cpu").detach().numpy()
        res = input1.grad
        res = res.to("cpu").numpy()
        return output, res

    def hardtanh_result(self, shape_format):
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 2)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
            cpu_output_forward, cpu_output_backward = self.cpu_op_backward_exec(cpu_input, 0, 1)
            npu_output_forward, npu_output_backward = self.npu_op_backward_exec(npu_input, 0, 1)
            cpu_output_forward = cpu_output_forward.astype(npu_output_forward.dtype)
            cpu_output_backward = cpu_output_backward.astype(npu_output_backward.dtype)
            self.assertRtolEqual(cpu_output_forward, npu_output_forward)
            self.assertRtolEqual(cpu_output_backward, npu_output_backward)

    def test_hardtanh_shape_format_fp16_1d(self, device="npu"):
        format_list = [0, 3, 4]
        shape_format = [
            [np.float16, i, [18]] for i in format_list
        ]
        self.hardtanh_result(shape_format)

    def test_hardtanh_shape_format_fp32_1d(self, device="npu"):
        format_list = [0, 3, 4]
        shape_format = [
            [np.float32, i, [18]] for i in format_list
        ]
        self.hardtanh_result(shape_format)

    def test_hardtanh_shape_format_fp16_2d(self, device="npu"):
        format_list = [0, 3, 4, 29]
        shape_format = [
            [np.float16, i, [256, 1000]] for i in format_list
        ]
        self.hardtanh_result(shape_format)

    def test_hardtanh_shape_format_fp32_2d(self, device="npu"):
        format_list = [0, 3, 4, 29]
        shape_format = [
            [np.float32, i, [256, 1000]] for i in format_list
        ]
        self.hardtanh_result(shape_format)

    def test_hardtanh_shape_format_fp16_3d(self, device="npu"):
        format_list = [0, 3, 4, 29]
        shape_format = [
            [np.float16, i, [32, 328, 368]] for i in format_list
        ]
        self.hardtanh_result(shape_format)

    def test_hardtanh_shape_format_fp32_3d(self, device="npu"):
        format_list = [0, 3, 4, 29]
        shape_format = [
            [np.float32, i, [32, 328, 368]] for i in format_list
        ]
        self.hardtanh_result(shape_format)

    def test_hardtanh_shape_format_fp16_4d(self, device="npu"):
        format_list = [0, 3, 4, 29]
        shape_format = [
            [np.float16, i, [256, 576, 7, 7]] for i in format_list
        ]
        self.hardtanh_result(shape_format)

    def test_hardtanh_shape_format_fp32_4d(self, device="npu"):
        format_list = [0, 3, 4, 29]
        shape_format = [
            [np.float32, i, [256, 576, 7, 7]] for i in format_list
        ]
        self.hardtanh_result(shape_format)


if __name__ == "__main__":
    run_tests()
