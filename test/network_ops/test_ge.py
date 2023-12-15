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


class TestGe(TestCase):
    def generate_scalar(self, min1, max1):
        scalar = np.random.uniform(min1, max1)
        return scalar

    def cpu_op_exec(self, input1, input2):
        output = torch.ge(input1, input2)
        output = output.numpy()
        return output

    def cpu_op_exec_out(self, input1, input2, input3):
        torch.ge(input1, input2, out=input3)
        output = input3.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.ge(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2, input3):
        torch.ge(input1, input2, out=input3)
        output = input3.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_inplace_exec(self, input1, input2):
        input1.ge_(input2)
        output = input1.numpy()
        return output

    def npu_op_inplace_exec(self, input1, input2):
        input1.ge_(input2)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_scalar(self, input1, scalar):
        output = torch.ge(input1, scalar)
        output = output.numpy()
        return output

    def cpu_op_exec_scalar_out(self, input1, scalar, input2):
        torch.ge(input1, scalar, out=input2)
        output = input2.numpy()
        return output

    def npu_op_exec_scalar(self, input1, scalar):
        output = torch.ge(input1, scalar)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_scalar_out(self, input1, scalar, input2):
        torch.ge(input1, scalar, out=input2)
        output = input2.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_inplace_exec_scalar(self, input1, scalar):
        input1.ge_(scalar)
        output = input1.numpy()
        return output

    def npu_op_inplace_exec_scalar(self, input1, scalar):
        input1.ge_(scalar)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def ge_tensor_out_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], -100, 100)
            cpu_input3 = torch.randn(item[1][2]) < 0
            npu_input3 = cpu_input3.npu()
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            if cpu_input2.dtype == torch.float16:
                cpu_input2 = cpu_input2.to(torch.float32)
            if cpu_input3.dtype == torch.float16:
                cpu_input3 = cpu_input3.to(torch.float32)
            cpu_output_out = self.cpu_op_exec_out(cpu_input1, cpu_input2, cpu_input3)
            npu_output_out = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3)
            cpu_output_out = cpu_output_out.astype(npu_output_out.dtype)
            self.assertRtolEqual(cpu_output_out, npu_output_out)

    def test_ge_tensor_out(self):
        shape_format = [
            [[np.float16, 0, [128, 116, 14, 14]], [np.float16, 0, [256, 116, 1, 1]]],
            [[np.float16, 0, [128, 3, 224, 224]], [np.float16, 0, [3, 3, 3]]],
            [[np.float16, 0, [128, 116, 14, 14]], [np.float16, 0, [128, 116, 14, 14]]],
            [[np.float32, 0, [256, 128, 7, 7]], [np.float32, 0, [128, 256, 3, 3]]],
            [[np.float32, 0, [2, 3, 3, 3]], [np.float32, 0, [3, 1, 3]]],
            [[np.float32, 0, [128, 232, 7, 7]], [np.float32, 0, [128, 232, 7, 7]]],
        ]
        self.ge_tensor_out_result(shape_format)

    def ge_scalar_out_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_input2 = torch.randn(item[1][2]) < 0
            npu_input2 = cpu_input2.npu()
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            if cpu_input2.dtype == torch.float16:
                cpu_input2 = cpu_input2.to(torch.float32)
            scalar = self.generate_scalar(0, 100)
            cpu_output_out = self.cpu_op_exec_scalar_out(cpu_input1, scalar, cpu_input2)
            npu_output_out = self.npu_op_exec_scalar_out(npu_input1, scalar, npu_input2)
            cpu_output_out = cpu_output_out.astype(npu_output_out.dtype)
            self.assertRtolEqual(cpu_output_out, npu_output_out)

    def test_ge_scalar_out(self):
        shape_format = [
            [[np.float16, 0, [4, 4, 128, 128]], [np.float16, 0, [256, 116, 1, 1]]],
            [[np.float16, 0, [12, 10, 14, 14]], [np.float16, 0, [256, 116, 1, 1]]],
            [[np.float16, 0, [16, 3, 1111, 1212]], [np.float16, 0, [3, 3, 3]]],
            [[np.float16, 0, [16, 16, 14, 14]], [np.float16, 0, [128, 116, 14, 14]]],
            [[np.float32, 0, [20, 10, 7, 7]], [np.float32, 0, [128, 256, 3, 3]]],
            [[np.float32, 0, [1313, 3, 3, 3]], [np.float32, 0, [3, 1, 3]]],
            [[np.float32, 0, [16, 22, 7, 7]], [np.float32, 0, [128, 232, 7, 7]]],
        ]
        self.ge_scalar_out_result(shape_format)

    def test_ge_bool(self):
        format_list = [0]
        shape_list = [(5, 3), (2, 3, 4)]
        scalar_list = [True, False]
        shape_format = [
            [[np.int32, i, j], k] for i in format_list for j in shape_list
            for k in scalar_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 0, 100)
            cpu_output1 = self.cpu_op_exec_scalar(cpu_input1 > 50, item[1])
            npu_output1 = self.npu_op_exec_scalar(npu_input1 > 50, item[1])
            cpu_output2 = self.cpu_op_exec(cpu_input1 > 50, cpu_input2 > 50)
            npu_output2 = self.npu_op_exec(npu_input1 > 50, npu_input2 > 50)

            self.assertRtolEqual(cpu_output1, npu_output1)
            self.assertRtolEqual(cpu_output2, npu_output2)

    def test_ge_scalar_float32(self):
        format_list = [0]
        shape_list = [(5, 3), (2, 3, 4)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            scalar = self.generate_scalar(0, 100)
            cpu_output = self.cpu_op_exec_scalar(cpu_input, scalar)
            npu_output = self.npu_op_exec_scalar(npu_input, scalar)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_ge_scalar_float16(self):
        format_list = [0]
        shape_list = [(5, 3), (2, 3, 4)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_input = cpu_input.to(torch.float32)
            scalar = self.generate_scalar(0, 100)
            cpu_output = self.cpu_op_exec_scalar(cpu_input, scalar)
            npu_output = self.npu_op_exec_scalar(npu_input, scalar)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_ge_scalar_int32(self):
        format_list = [0]
        shape_list = [(5, 3), (2, 3, 4)]
        shape_format = [
            [np.int32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            scalar = self.generate_scalar(0, 100)
            cpu_output = self.cpu_op_exec_scalar(cpu_input, scalar)
            npu_output = self.npu_op_exec_scalar(npu_input, scalar)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_ge_tensor_float32(self):
        format_list = [0]
        shape_list = [(5, 3), (2, 3, 4)]
        shape_format = [[[np.float32, i, j], [np.float32, i, j]]
                        for i in format_list for j in shape_list]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_ge_tensor_float16(self):
        format_list = [0]
        shape_list = [(5, 3), (2, 3, 4)]
        shape_format = [[[np.float16, i, j], [np.float16, i, j]]
                        for i in format_list for j in shape_list]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_ge_inplace_float32(self):
        format_list = [0]
        shape_list = [(5, 3), (2, 3, 4)]
        shape_format = [[[np.float32, i, j], [np.float32, i, j]]
                        for i in format_list for j in shape_list]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            cpu_output = self.cpu_op_inplace_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_inplace_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_ge_inplace_float16(self):
        format_list = [0, 3]
        shape_list = [(5, 3), (2, 3, 4)]
        shape_format = [[[np.float16, i, j], [np.float16, i, j]]
                        for i in format_list for j in shape_list]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_inplace_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_inplace_exec(npu_input1, npu_input2)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_ge_inplace_scalar_float32(self):
        format_list = [0]
        shape_list = [(5, 3), (2, 3, 4)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            scalar = self.generate_scalar(0, 100)
            cpu_output = self.cpu_op_inplace_exec_scalar(cpu_input, scalar)
            npu_output = self.npu_op_inplace_exec_scalar(npu_input, scalar)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_ge_inplace_scalar_float16(self):
        format_list = [0]
        shape_list = [(5, 3), (2, 3, 4)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_input = cpu_input.to(torch.float32)
            scalar = self.generate_scalar(0, 100)
            cpu_output = self.cpu_op_inplace_exec_scalar(cpu_input, scalar)
            npu_output = self.npu_op_inplace_exec_scalar(npu_input, scalar)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_ge_mix_dtype(self):
        cpu_input1, npu_input1 = create_common_tensor([np.float16, 0, (2, 3)], 1, 100)
        cpu_input2, npu_input2 = create_common_tensor([np.float32, 0, (2, 3)], 1, 100)
        cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_ge_diff_device(self):
        input1 = cmp1 = torch.randn(5, 5)
        input2 = cmp2 = torch.tensor(1)
        diff_device_out = torch.ge(input1.to('npu'), input2)
        diff_device_cmp = torch.ge(cmp1, cmp2)
        self.assertRtolEqual(diff_device_out, diff_device_cmp)

        input1 = cmp1 = torch.tensor(1)
        input2 = cmp2 = torch.tensor(2)
        diff_device_out = torch.ge(input1, input2.to('npu'))
        diff_device_cmp = torch.ge(cmp1, cmp2)
        self.assertRtolEqual(diff_device_out, diff_device_cmp)


if __name__ == '__main__':
    run_tests()
