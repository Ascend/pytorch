# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION.
# All rights reserved.
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


class TestCummin(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input_x = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input = torch.from_numpy(input_x)
        return npu_input

    def generate_dimname_data(self, min_d, max_d, shape, dtype):
        input_x = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input = torch.from_numpy(input_x)
        npu_input.names = ['N', 'C', 'H', 'W']
        return npu_input

    def cpu_op_exec(self, input_x, dim):
        output, argmin = torch.cummin(input_x, dim)
        output = output.numpy()
        argmin = argmin.numpy().astype(np.int32)
        return output, argmin

    def npu_op_exec(self, input_x, dim):
        input1 = input_x.to("npu")
        output, argmin = torch.cummin(input1, dim)
        output = output.to("cpu")
        output = output.numpy()
        argmin = argmin.to("cpu")
        argmin = argmin.numpy().astype(np.int32)
        return output, argmin

    def npu_op_exec_out(self, input_x, dim, output_value, output_argmin):
        input_x = input_x.to("npu")
        output_value = output_value.to("npu")
        output_argmin = output_argmin.to("npu").to(torch.long)
        torch.cummin(input_x, dim, out=(output_value, output_argmin))
        output_value = output_value.to("cpu")
        output_value = output_value.numpy()
        output_argmin = output_argmin.to("cpu")
        output_argmin = output_argmin.numpy().astype(np.int32)
        return output_value, output_argmin

    def test_cummin_dim2_0_float32(self):
        input_x1 = self.generate_data(-1, 1, (3, 3), np.float32)
        cpu_output, cpu_argmin = self.cpu_op_exec(input_x1, 1)
        npu_output, npu_argmin = self.npu_op_exec(input_x1, 1)
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpu_argmin, npu_argmin)

    def test_cummin_dim6_4_float32(self):
        input_x1 = self.generate_data(-1, 1, (3, 3, 3, 3, 3, 3), np.float32)
        cpu_output, cpu_argmin = self.cpu_op_exec(input_x1, 4)
        npu_output, npu_argmin = self.npu_op_exec(input_x1, 4)
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpu_argmin, npu_argmin)

    def test_cummin_dim2_2_int32(self):
        input_x1 = self.generate_data(-1, 1, (3, 3), np.int32)
        cpu_output, cpu_argmin = self.cpu_op_exec(input_x1, 1)
        npu_output, npu_argmin = self.npu_op_exec(input_x1, 1)
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpu_argmin, npu_argmin)

    def test_cummin_dim2_2_int32_out(self):
        input_x1 = self.generate_data(-1, 1, (3, 3), np.int32)
        output_values = self.generate_data(-1, 1, (3, 3), np.int32)
        output_argmin = self.generate_data(-1, 1, (3, 3), np.int32)
        cpu_output, cpu_argmin = self.cpu_op_exec(input_x1, 1)
        npu_output, npu_argmin = self.npu_op_exec_out(input_x1, 1, output_values, output_argmin)
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpu_argmin, npu_argmin)

    def test_cummin_dim6_5_float32(self):
        input_x1 = self.generate_data(-1, 1, (3, 3, 3, 3, 3, 3), np.float32)
        cpu_output, cpu_argmin = self.cpu_op_exec(input_x1, 5)
        npu_output, npu_argmin = self.npu_op_exec(input_x1, 5)
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpu_argmin, npu_argmin)

    def test_cummin_dim2_1_out_float32(self):
        input_x1 = self.generate_data(-1, 1, (3, 3), np.float32)
        output_values = self.generate_data(-1, 1, (3, 3), np.float32)
        output_argmin = self.generate_data(-1, 1, (3, 3), np.int32)
        cpu_output, cpu_argmin = self.cpu_op_exec(input_x1, 1)
        npu_output, npu_argmin = self.npu_op_exec_out(input_x1, 1, output_values, output_argmin)
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpu_argmin, npu_argmin)

    def test_cummin_dim5_2_out_float32(self):
        input_x1 = self.generate_data(-1, 1, (3, 3, 3, 3, 3), np.float32)
        output_values = self.generate_data(-1, 1, (3, 3, 3, 3, 3), np.float32)
        output_argmin = self.generate_data(-1, 1, (3, 3, 3, 3, 3), np.int32)
        cpu_output, cpu_argmin = self.cpu_op_exec(input_x1, 2)
        npu_output, npu_argmin = self.npu_op_exec_out(input_x1, 2, output_values, output_argmin)
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpu_argmin, npu_argmin)

    def test_cummin_10dim6_2_float32(self):
        input_x1 = self.generate_data(-1, 1, (10, 10, 10, 10, 10, 10), np.float32)
        cpu_output, cpu_argmin = self.cpu_op_exec(input_x1, 2)
        npu_output, npu_argmin = self.npu_op_exec(input_x1, 2)
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpu_argmin, npu_argmin)

    def test_cummin_dim4_N_out_float32_dimname(self):
        input_x1 = self.generate_dimname_data(-1, 1, (3, 3, 3, 3), np.float32)
        output_values = self.generate_data(-1, 1, (3, 3, 3, 3), np.float32)
        output_argmin = self.generate_data(-1, 1, (3, 3, 3, 3), np.int32)
        cpu_output, cpu_argmin = self.cpu_op_exec(input_x1, 'N')
        npu_output, npu_argmin = self.npu_op_exec_out(input_x1, 'N', output_values, output_argmin)
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpu_argmin, npu_argmin)

    def test_cummin_dim4_H_float32_dimname(self):
        input_x1 = self.generate_dimname_data(-1, 1, (3, 3, 3, 3), np.float32)
        cpu_output, cpu_argmin = self.cpu_op_exec(input_x1, 'H')
        npu_output, npu_argmin = self.npu_op_exec(input_x1, 'H')
        self.assertRtolEqual(cpu_output, npu_output)
        self.assertRtolEqual(cpu_argmin, npu_argmin)


if __name__ == "__main__":
    torch.npu.set_device("npu:0")
    run_tests()
