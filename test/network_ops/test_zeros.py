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

import copy
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestZeros(TestCase):
    def cpu_op_exec(self, input1, dtype):
        output = torch.zeros(input1.size(), dtype=dtype, device="cpu")
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, dtype):
        output = torch.zeros(input1.size(), dtype=dtype, device="npu")
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2, dtype):
        torch.zeros(input1.size(), dtype=dtype, device="npu", out=input2)
        output = input2.to("cpu")
        output = output.numpy()
        return output

    def zeros_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            npu_input2 = copy.deepcopy(cpu_input1)
            npu_input2 = npu_input2.to(item[1]).to('npu')
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)

            cpu_output = self.cpu_op_exec(cpu_input1, item[1])
            npu_output = self.npu_op_exec(npu_input1, item[1])
            npu_output_out = self.npu_op_exec_out(npu_input1, npu_input2, item[1])
            cpu_output = cpu_output.astype(npu_output.dtype)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output, npu_output_out)

    def test_zeros_shape_format_names(self, device="npu"):
        format_list = [0, 3, 29]
        dtype_list = [torch.float16, torch.float32, torch.int32]
        shape_format = [[[np.float32, i, [18, 24, 8, 8]], j] for i in format_list for j in dtype_list]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)

            cpu_output = torch.zeros(cpu_input1.size(), names=('N', 'C', 'H', 'W'), dtype=item[1], device="cpu")
            cpu_output = cpu_output.numpy()
            npu_output = torch.zeros(cpu_input1.size(), names=('N', 'C', 'H', 'W'), dtype=item[1], device="npu")
            npu_output = npu_output.to("cpu")
            npu_output = npu_output.numpy()
            cpu_output = cpu_output.astype(npu_output.dtype)

            self.assertRtolEqual(cpu_output, npu_output)

    def test_zeros_shape_format_fp16_1d(self, device="npu"):
        format_list = [0, 3, 29]
        dtype_list = [torch.float16, torch.float32, torch.int32]
        shape_format = [[[np.float16, i, [18]], j] for i in format_list for j in dtype_list]
        self.zeros_result(shape_format)

    def test_zeros_shape_format_fp16_2d(self, device="npu"):
        format_list = [0, 3, 29]
        dtype_list = [torch.float16, torch.float32, torch.int32]
        shape_format = [[[np.float16, i, [5, 256]], j] for i in format_list for j in dtype_list]
        self.zeros_result(shape_format)

    def test_zeros_shape_format_fp16_3d(self, device="npu"):
        format_list = [0, 3, 29]
        dtype_list = [torch.float16, torch.float32, torch.int32]
        shape_format = [[[np.float16, i, [32, 3, 3]], j] for i in format_list for j in dtype_list]
        self.zeros_result(shape_format)

    def test_zeros_shape_format_fp16_4d(self, device="npu"):
        format_list = [0, 3, 29]
        dtype_list = [torch.float16, torch.float32, torch.int32]
        shape_format = [[[np.float16, i, [64, 112, 7, 7]], j] for i in format_list for j in dtype_list]
        self.zeros_result(shape_format)

    def test_zeros_shape_format_fp32_1d(self, device="npu"):
        format_list = [0, 3, 29]
        dtype_list = [torch.float16, torch.float32, torch.int32]
        shape_format = [[[np.float32, i, [18]], j] for i in format_list for j in dtype_list]
        self.zeros_result(shape_format)

    def test_zeros_shape_format_fp32_2d(self, device="npu"):
        format_list = [0, 3, 29]
        dtype_list = [torch.float16, torch.float32, torch.int32]
        shape_format = [[[np.float32, i, [5, 256]], j] for i in format_list for j in dtype_list]
        self.zeros_result(shape_format)

    def test_zeros_shape_format_fp32_3d(self, device="npu"):
        format_list = [0, 3, 29]
        dtype_list = [torch.float16, torch.float32, torch.int32]
        shape_format = [[[np.float32, i, [32, 3, 3]], j] for i in format_list for j in dtype_list]
        self.zeros_result(shape_format)

    def test_zeros_shape_format_fp32_4d(self, device="npu"):
        format_list = [0, 3, 29]
        dtype_list = [torch.float16, torch.float32, torch.int32]
        shape_format = [[[np.float32, i, [64, 112, 7, 7]], j] for i in format_list for j in dtype_list]
        self.zeros_result(shape_format)

    def test_zeros_shape_format_int32_1d(self, device="npu"):
        format_list = [-1, 0]
        dtype_list = [torch.float16, torch.float32, torch.int32]
        shape_format = [[[np.int32, i, [18]], j] for i in format_list for j in dtype_list]
        self.zeros_result(shape_format)

    def test_zeros_shape_format_int32_2d(self, device="npu"):
        format_list = [-1, 0]
        dtype_list = [torch.float16, torch.float32, torch.int32]
        shape_format = [[[np.int32, i, [5, 256]], j] for i in format_list for j in dtype_list]
        self.zeros_result(shape_format)

    def test_zeros_shape_format_int32_3d(self, device="npu"):
        format_list = [-1, 0]
        dtype_list = [torch.float16, torch.float32, torch.int32]
        shape_format = [[[np.int32, i, [32, 3, 3]], j] for i in format_list for j in dtype_list]
        self.zeros_result(shape_format)

    def test_zeros_shape_format_int32_4d(self, device="npu"):
        format_list = [-1, 0]
        dtype_list = [torch.float16, torch.float32, torch.int32]
        shape_format = [[[np.int32, i, [64, 112, 7, 7]], j] for i in format_list for j in dtype_list]
        self.zeros_result(shape_format)


if __name__ == "__main__":
    run_tests()
