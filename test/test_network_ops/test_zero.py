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


class TestZero(TestCase):
    def cpu_op_exec(self, input1):
        torch.zero_(input1)
        output = input1.numpy()
        return output

    def npu_op_exec(self, input1):
        torch.zero_(input1)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def zero_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)

            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)

            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_zero_shape_format_fp16_1d(self, device="npu"):
        format_list = [0, 3, 29]
        shape_format = [[np.float16, i, [18]] for i in format_list]
        self.zero_result(shape_format)

    def test_zero_shape_format_fp16_2d(self, device="npu"):
        format_list = [0, 3, 29]
        shape_format = [[np.float16, i, [5, 256]] for i in format_list]
        self.zero_result(shape_format)

    def test_zero_shape_format_fp16_3d(self, device="npu"):
        format_list = [0, 3, 29]
        shape_format = [[np.float16, i, [32, 3, 3]] for i in format_list]
        self.zero_result(shape_format)

    def test_zero_shape_format_fp16_4d(self, device="npu"):
        format_list = [0, 3, 29]
        shape_format = [[np.float16, i, [64, 112, 7, 7]] for i in format_list]
        self.zero_result(shape_format)

    def test_zero_shape_format_fp32_1d(self, device="npu"):
        format_list = [0, 3, 29]
        shape_format = [[np.float32, i, [18]] for i in format_list]
        self.zero_result(shape_format)

    def test_zero_shape_format_fp32_2d(self, device="npu"):
        format_list = [0, 3, 29]
        shape_format = [[np.float32, i, [5, 256]] for i in format_list]
        self.zero_result(shape_format)

    def test_zero_shape_format_fp32_3d(self, device="npu"):
        format_list = [0, 3, 29]
        shape_format = [[np.float32, i, [32, 3, 3]] for i in format_list]
        self.zero_result(shape_format)

    def test_zero_shape_format_fp32_4d(self, device="npu"):
        format_list = [0, 3, 29]
        shape_format = [[np.float32, i, [64, 112, 7, 7]] for i in format_list]
        self.zero_result(shape_format)

    def test_zero_shape_format_int32_1d(self, device="npu"):
        format_list = [-1, 0]
        shape_format = [[np.int32, i, [18]] for i in format_list]
        self.zero_result(shape_format)

    def test_zero_shape_format_int32_2d(self, device="npu"):
        format_list = [-1, 0]
        shape_format = [[np.int32, i, [5, 256]] for i in format_list]
        self.zero_result(shape_format)

    def test_zero_shape_format_int32_3d(self, device="npu"):
        format_list = [-1, 0]
        shape_format = [[np.int32, i, [32, 3, 3]] for i in format_list]
        self.zero_result(shape_format)

    def test_zero_shape_format_int32_4d(self, device="npu"):
        format_list = [-1, 0]
        shape_format = [[np.int32, i, [64, 112, 7, 7]] for i in format_list]
        self.zero_result(shape_format)


if __name__ == "__main__":
    run_tests()
