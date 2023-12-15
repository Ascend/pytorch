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


class TestAddmm(TestCase):

    def generate_scalar(self, dtype, min_x, max_x):
        if dtype == "float32" or "float16":
            scalar = np.random.uniform(min_x, max_x)
        if dtype == "int32":
            scalar = np.random.randint(min_x, max_x)
        return scalar

    def cpu_op_exec(self, input1, input2, input3, scalar1, scalar2):
        output = torch.addmm(
            input1,
            input2,
            input3,
            beta=scalar1,
            alpha=scalar2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2, input3, scalar1, scalar2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        input3 = input3.to("npu")
        output = torch.addmm(
            input1,
            input2,
            input3,
            beta=scalar1,
            alpha=scalar2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(
            self,
            input1,
            input2,
            input3,
            scalar1,
            scalar2,
            input4):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        input3 = input3.to("npu")
        output = input4.to("npu")
        torch.addmm(
            input1,
            input2,
            input3,
            beta=scalar1,
            alpha=scalar2,
            out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_inplace(self, input1, input2, input3, scalar1, scalar2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        input3 = input3.to("npu")
        input1.addmm_(input2, input3, beta=scalar1, alpha=scalar2)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_transpose_exec(self, input1, input2, input3, scalar1, scalar2):
        input3_t = input3.t()
        output = torch.addmm(
            input1,
            input2,
            input3_t,
            beta=scalar1,
            alpha=scalar2)
        output = output.numpy()
        return output

    def npu_op_transpose_exec(self, input1, input2, input3, scalar1, scalar2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        input3 = input3.to("npu")
        input3_t = input3.t()
        output = torch.addmm(
            input1,
            input2,
            input3_t,
            beta=scalar1,
            alpha=scalar2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_addmm_shape_format_int(self):
        format_list = [0]
        shape_list = [(3, 3), (3, 5), (5, 3)]
        shape_format1 = [
            [np.int32, i, shape_list[0]] for i in format_list
        ]
        shape_format2 = [
            [np.int32, i, shape_list[1]] for i in format_list
        ]
        shape_format3 = [
            [np.int32, i, shape_list[2]] for i in format_list
        ]
        shape_format = [[i, j, k, "int32"]
                        for i in shape_format1 for j in shape_format2 for k in shape_format3]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            cpu_input3, npu_input3 = create_common_tensor(item[2], 0, 100)
            cpu_input4, npu_input4 = create_common_tensor(item[0], 0, 100)

            scalar1 = self.generate_scalar(item[3], 0, 10)
            scalar2 = self.generate_scalar(item[3], 0, 10)

            cpu_output = self.cpu_op_exec(
                cpu_input1, cpu_input2, cpu_input3, scalar1, scalar2)
            npu_output = self.npu_op_exec(
                npu_input1, npu_input2, npu_input3, scalar1, scalar2)

            npu_output1 = self.npu_op_exec_out(
                npu_input1, npu_input2, npu_input3, scalar1, scalar2, npu_input4)
            npu_output2 = self.npu_op_exec_inplace(
                npu_input1, npu_input2, npu_input3, scalar1, scalar2)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output, npu_output1)
            self.assertRtolEqual(cpu_output, npu_output2)

    def test_addmm_shape_format_fp32(self):
        format_list = [0]
        shape_list = [(3, 3), (3, 5), (5, 3)]
        shape_format1 = [
            [np.float16, i, shape_list[0]] for i in format_list
        ]
        shape_format2 = [
            [np.float16, i, shape_list[1]] for i in format_list
        ]
        shape_format3 = [
            [np.float16, i, shape_list[2]] for i in format_list
        ]
        shape_format = [[i, j, k, "float32"]
                        for i in shape_format1 for j in shape_format2 for k in shape_format3]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 1)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 1)
            cpu_input3, npu_input3 = create_common_tensor(item[2], 0, 1)
            cpu_input4, npu_input4 = create_common_tensor(item[0], 0, 1)

            scalar1 = self.generate_scalar(item[3], 0, 2)
            scalar2 = self.generate_scalar(item[3], 0, 2)

            cpu_output = self.cpu_op_exec(
                cpu_input1.float(), cpu_input2.float(), cpu_input3.float(), scalar1, scalar2)
            npu_output = self.npu_op_exec(
                npu_input1.float(), npu_input2.float(), npu_input3.float(), scalar1, scalar2)

            npu_output1 = self.npu_op_exec_out(
                npu_input1.float(), npu_input2.float(), npu_input3.float(), scalar1, scalar2, npu_input4.float())
            npu_output2 = self.npu_op_exec_inplace(
                npu_input1.float(), npu_input2.float(), npu_input3.float(), scalar1, scalar2)

            self.assertRtolEqual(cpu_output, npu_output, prec=1.e-3, prec16=1.e-3)
            self.assertRtolEqual(cpu_output, npu_output1, prec=1.e-3, prec16=1.e-3)
            self.assertRtolEqual(cpu_output, npu_output2, prec=1.e-3, prec16=1.e-3)

    def test_addmm_shape_format_fp16(self):
        format_list = [0]
        shape_list = [(3, 3), (3, 5), (5, 3)]
        shape_format1 = [
            [np.float16, i, shape_list[0]] for i in format_list
        ]
        shape_format2 = [
            [np.float16, i, shape_list[1]] for i in format_list
        ]
        shape_format3 = [
            [np.float16, i, shape_list[2]] for i in format_list
        ]
        shape_format = [[i, j, k, "float16"]
                        for i in shape_format1 for j in shape_format2 for k in shape_format3]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 2)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 2)
            cpu_input3, npu_input3 = create_common_tensor(item[2], 0, 2)
            cpu_input4, npu_input4 = create_common_tensor(item[0], 0, 2)

            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_input2 = cpu_input2.to(torch.float32)
            cpu_input3 = cpu_input3.to(torch.float32)

            scalar1 = self.generate_scalar(item[3], 0, 10)
            scalar2 = self.generate_scalar(item[3], 0, 10)

            cpu_output = self.cpu_op_exec(
                cpu_input1, cpu_input2, cpu_input3, scalar1, scalar2)
            npu_output = self.npu_op_exec(
                npu_input1, npu_input2, npu_input3, scalar1, scalar2)
            cpu_output = cpu_output.astype(npu_output.dtype)

            npu_output1 = self.npu_op_exec_out(
                npu_input1, npu_input2, npu_input3, scalar1, scalar2, npu_input4)
            npu_output2 = self.npu_op_exec_inplace(
                npu_input1, npu_input2, npu_input3, scalar1, scalar2)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output, npu_output1)
            self.assertRtolEqual(cpu_output, npu_output2)

    def test_addmm_transpose_shape_format_int(self):
        format_list = [0]
        shape_list = [(4, 5), (4, 7), (5, 7)]
        shape_format1 = [
            [np.int32, i, shape_list[0]] for i in format_list
        ]
        shape_format2 = [
            [np.int32, i, shape_list[1]] for i in format_list
        ]
        shape_format3 = [
            [np.int32, i, shape_list[2]] for i in format_list
        ]
        shape_format = [[i, j, k, "int32"]
                        for i in shape_format1 for j in shape_format2 for k in shape_format3]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            cpu_input3, npu_input3 = create_common_tensor(item[2], 0, 100)

            scalar1 = self.generate_scalar(item[3], 0, 10)
            scalar2 = self.generate_scalar(item[3], 0, 10)

            cpu_transpose_output = self.cpu_op_transpose_exec(
                cpu_input1, cpu_input2, cpu_input3, scalar1, scalar2)
            npu_transpose_output = self.npu_op_transpose_exec(
                npu_input1, npu_input2, npu_input3, scalar1, scalar2)

            self.assertRtolEqual(cpu_transpose_output, npu_transpose_output)

    def test_addmm_transpose_shape_format_fp32(self):
        format_list = [0]
        shape_list = [(4, 5), (4, 7), (5, 7)]
        shape_format1 = [
            [np.float16, i, shape_list[0]] for i in format_list
        ]
        shape_format2 = [
            [np.float16, i, shape_list[1]] for i in format_list
        ]
        shape_format3 = [
            [np.float16, i, shape_list[2]] for i in format_list
        ]
        shape_format = [[i, j, k, "float32"]
                        for i in shape_format1 for j in shape_format2 for k in shape_format3]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 1)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 1)
            cpu_input3, npu_input3 = create_common_tensor(item[2], 0, 1)

            scalar1 = self.generate_scalar(item[3], 0, 2)
            scalar2 = self.generate_scalar(item[3], 0, 2)

            cpu_transpose_output = self.cpu_op_transpose_exec(
                cpu_input1.float(), cpu_input2.float(), cpu_input3.float(), scalar1, scalar2)
            npu_transpose_output = self.npu_op_transpose_exec(
                npu_input1.float(), npu_input2.float(), npu_input3.float(), scalar1, scalar2)

            self.assertRtolEqual(cpu_transpose_output, npu_transpose_output, prec=1.e-3, prec16=1.e-3)

    def test_addmm_transpose_shape_format_fp16(self):
        format_list = [0]
        shape_list = [(4, 5), (4, 7), (5, 7)]
        shape_format1 = [
            [np.float16, i, shape_list[0]] for i in format_list
        ]
        shape_format2 = [
            [np.float16, i, shape_list[1]] for i in format_list
        ]
        shape_format3 = [
            [np.float16, i, shape_list[2]] for i in format_list
        ]
        shape_format = [[i, j, k, "float16"]
                        for i in shape_format1 for j in shape_format2 for k in shape_format3]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 2)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 2)
            cpu_input3, npu_input3 = create_common_tensor(item[2], 0, 2)

            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_input2 = cpu_input2.to(torch.float32)
            cpu_input3 = cpu_input3.to(torch.float32)

            scalar1 = self.generate_scalar(item[3], 0, 10)
            scalar2 = self.generate_scalar(item[3], 0, 10)

            cpu_transpose_output = self.cpu_op_transpose_exec(
                cpu_input1, cpu_input2, cpu_input3, scalar1, scalar2)
            npu_transpose_output = self.npu_op_transpose_exec(
                npu_input1, npu_input2, npu_input3, scalar1, scalar2)
            cpu_transpose_output = cpu_transpose_output.astype(
                npu_transpose_output.dtype)

            self.assertRtolEqual(cpu_transpose_output, npu_transpose_output)


if __name__ == "__main__":
    run_tests()
