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
import copy
import sys
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestLe(TestCase):
    def generate_scalar(self, min, max):
        scalar = np.random.uniform(min, max)
        return scalar

    def cpu_op_exec(self, input1, input2):
        output = torch.le(input1, input2)
        output = output.numpy()
        return output

    def cpu_op_exec_out(self, input1, input2, input3):
        torch.le(input1, input2, out = input3)
        output = input3.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.le(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_inplace_exec(self, input1, input2):
        output = input1.le_(input2)
        output = input1
        output = output.numpy()
        return output

    def npu_op_inplace_exec(self, input1, input2):
        output = input1.le_(input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2, output):
        torch.le(input1, input2, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_scalar(self, input, scalar):
        output = torch.le(input, scalar)
        output = output.numpy()
        return output

    def cpu_op_exec_scalar_out(self, input1, scalar, input2):
        torch.le(input1, scalar, out = input2)
        output = input2.numpy()
        return output

    def npu_op_exec_scalar(self, input, scalar):
        output = torch.le(input, scalar)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_inplace_exec_scalar(self, input, scalar):
        output = input.le_(scalar)
        output = output.numpy()
        return output

    def npu_op_inplace_exec_scalar(self, input, scalar):
        input = input.to("npu")
        output = input.le_(scalar)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_scalar_out(self, input, scalar, output):
        torch.le(input, scalar, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_inplace_stride_exec(self, input1, input2):
        input1 = input1.as_strided([2, 2], [1, 2], 1)
        input2 = input2.as_strided([2, 2], [1, 2], 1)
        output = input1.le_(input2)
        output = output.numpy()
        return output

    def npu_op_inplace_stride_exec(self, input1, input2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        input1 = input1.as_strided([2, 2], [1, 2], 1)
        input2 = input2.as_strided([2, 2], [1, 2], 1)
        output = input1.le_(input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_inplace_stride_scalar_exec(self, input1, input2):
        input1 = input1.as_strided([2, 2], [1, 2], 1)
        output = input1.le_(input2)
        output = output.numpy()
        return output

    def npu_op_inplace_stride_scalar_exec(self, input1, input2):
        input1 = input1.to("npu")
        input1 = input1.as_strided([2, 2], [1, 2], 1)
        output = input1.le_(input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def le_tensor_out_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], -100, 100)
            cpu_input3 = torch.randn(item[1][2])<0
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

    def test_le_tensor_out(self, device):
        shape_format = [
            [[np.float16, 0, [128, 116, 14, 14]], [np.float16, 0, [256, 116, 1, 1]]],
            [[np.float16, 0, [128, 3, 224, 224]], [np.float16, 0, [3, 3, 3]]],
            [[np.float16, 0, [128, 116, 14, 14]], [np.float16, 0, [128, 116, 14, 14]]],
            [[np.float32, 0, [256, 128, 7, 7]],   [np.float32, 0, [128, 256, 3, 3]]],
            [[np.float32, 0, [2, 3, 3, 3]],       [np.float32, 0, [3, 1, 3]]],
            [[np.float32, 0, [128, 232, 7, 7]],   [np.float32, 0, [128, 232, 7, 7]]],
        ]
        self.le_tensor_out_result(shape_format)

    def le_scalar_out_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_input2 = torch.randn(item[1][2])<0
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

    def test_le_scalar_out(self, device):
        shape_format = [
            [[np.float16, 0, [12, 4, 12, 121]], [np.float16, 0, [256, 116, 1, 1]]],
            [[np.float16, 0, [12, 10, 14, 111]], [np.float16, 0, [256, 116, 1, 1]]],
            [[np.float16, 2, [16, 3, 11, 121, 21]], [np.float16, 0, [3, 3, 3]]],
            [[np.float16, 0, [16, 16, 14]], [np.float16, 0, [128, 116, 14, 14]]],
            [[np.float32, 0, [20, 10, 7, 7]], [np.float32, 0, [128, 256, 3, 3]]],
            [[np.float32, 2, [1313, 3, 3, 3, 121]], [np.float32, 0, [3, 1, 3]]],
            [[np.float32, 0, [16, 22, 7, 7]], [np.float32, 0, [128, 232, 7, 7]]],
        ]
        self.le_scalar_out_result(shape_format)

    def test_le_scalar_float32(self, device):
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
            self.assertEqual(cpu_output, npu_output)

    def test_le_scalar_int32(self, device):
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
            self.assertEqual(cpu_output, npu_output)

    def test_gt_scalar_float16(self, device):
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
            cpu_output = cpu_output.astype(np.float16)
            self.assertEqual(cpu_output, npu_output)

    def test_le_tensor_float32(self, device):
        format_list = [0]
        shape_list = [(5, 3), (2, 3, 4)]
        shape_format = [[[np.float32, i, j], [np.float32, i, j]]
                        for i in format_list for j in shape_list]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertEqual(cpu_output, npu_output)

    def test_le_tensor_float16(self, device):
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
            cpu_output = cpu_output.astype(np.float16)
            self.assertEqual(cpu_output, npu_output)

    def test_le_inplace_float32(self, device):
        format_list = [0, 3]
        shape_list = [(5, 3), (2, 3, 4)]
        shape_format = [[[np.float32, i, j], [np.float32, i, j]]
                        for i in format_list for j in shape_list]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            cpu_output = self.cpu_op_inplace_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_inplace_exec(npu_input1, npu_input2)
            self.assertEqual(cpu_output, npu_output)

    def test_le_inplace_float16(self, device):
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
            cpu_output = cpu_output.astype(np.float16)
            self.assertEqual(cpu_output, npu_output)

    def test_le_inplace_scalar_float32(self, device):
        format_list = [0]
        shape_list = [(5, 3), (2, 3, 4)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            scalar = self.generate_scalar(0, 100)
            scalar1 = copy.deepcopy(scalar)
            ncpu_input = copy.deepcopy(cpu_input)
            cpu_output = self.cpu_op_inplace_exec_scalar(cpu_input, scalar)
            npu_output = self.npu_op_inplace_exec_scalar(npu_input, scalar1)
            self.assertEqual(cpu_output, npu_output)

    def test_le_inplace_scalar_float16(self, device):
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
            cpu_output = cpu_output.astype(np.float16)
            self.assertEqual(cpu_output, npu_output)

    def test_le_mix_dtype(self, device):
        npu_input1, npu_input2 = create_common_tensor([np.float16, 0, (2, 3)], 1, 100)
        npu_input3, npu_input4 = create_common_tensor([np.float32, 0, (2, 3)], 1, 100)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input3)
        npu_output = self.npu_op_exec(npu_input2, npu_input4)
        self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestLe, globals(), except_for="cpu")
if __name__ == '__main__':
    run_tests()
