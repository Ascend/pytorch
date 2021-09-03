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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestNotEqual(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = torch.ne(input1, input2)
        output = output.numpy().astype(np.int32)
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.ne(input1, input2)
        output = output.to("cpu")
        output = output.numpy().astype(np.int32)
        return output

    def cpu_op_inplace_exec(self, input1, input2):
        input1.ne_(input2)
        output = input1.numpy().astype(np.int32)
        return output

    def npu_op_inplace_exec(self, input1, input2):
        input1.ne_(input2)
        output = input1.to("cpu")
        output = output.numpy().astype(np.int32)
        return output

    def npu_op_exec_out(self, input1, input2, out):
        torch.ne(input1, input2, out=out)
        output = out.to("cpu")
        output = output.numpy().astype(np.int32)
        return output

    def not_equal_scalar_result(self, shape_format):
        for item in shape_format:
            scalar = np.random.uniform(0, 100)
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            npu_input3 = copy.deepcopy(cpu_input1).to("npu").to(torch.bool)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)

            cpu_output = self.cpu_op_exec(cpu_input1, scalar)
            npu_output = self.npu_op_exec(npu_input1, scalar)
            npu_output_out = self.npu_op_exec_out(npu_input1, scalar, npu_input3)

            cpu_output_inp = self.cpu_op_inplace_exec(cpu_input1, scalar)
            npu_output_inp = self.npu_op_inplace_exec(npu_input1, scalar)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output, npu_output_out)
            self.assertRtolEqual(cpu_output_inp, npu_output_inp)

    def not_equal_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            npu_input3 = copy.deepcopy(cpu_input1).to("npu").to(torch.bool)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            npu_output_out = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3)

            cpu_output_inp = self.cpu_op_inplace_exec(cpu_input1, cpu_input2)
            npu_output_inp = self.npu_op_inplace_exec(npu_input1, npu_input2)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output, npu_output_out)
            self.assertRtolEqual(cpu_output_inp, npu_output_inp)

    def test_not_equal_shape_format_fp16_1d(self, device):
        format_list = [-1, 0]
        shape_format = [[[np.float16, i, [16]], [np.float16, i, [16]]] for i in format_list]
        self.not_equal_result(shape_format)

    def test_not_equal_shape_format_fp32_1d(self, device):
        format_list = [-1, 0]
        shape_format = [[[np.float32, i, [16]], [np.float32, i, [16]]] for i in format_list]
        self.not_equal_result(shape_format)

    def test_not_equal_shape_format_fp16_2d(self, device):
        format_list = [-1, 0]
        shape_format = [[[np.float16, i, [448, 1]], [np.float16, i, [448, 1]]] for i in format_list]
        self.not_equal_result(shape_format)

    def test_not_equal_shape_format_fp32_2d(self, device):
        format_list = [-1, 0]
        shape_format = [[[np.float32, i, [448, 1]], [np.float32, i, [448, 1]]] for i in format_list]
        self.not_equal_result(shape_format)

    def test_not_equal_shape_format_fp16_3d(self, device):
        format_list = [-1, 0]
        shape_format = [[[np.float16, i, [16, 640, 640]], [np.float16, i, [16, 640, 640]]] for i in format_list]
        self.not_equal_result(shape_format)

    def test_not_equal_shape_format_fp32_3d(self, device):
        format_list = [-1, 0, 3]
        shape_format = [[[np.float32, i, [16, 640, 640]], [np.float32, i, [16, 640, 640]]] for i in format_list]
        self.not_equal_result(shape_format)

    def test_not_equal_shape_format_fp16_4d(self, device):
        format_list = [-1, 0, 3]
        shape_format = [[[np.float16, i, [32, 3, 3, 3]], [np.float16, i, [32, 3, 3, 3]]] for i in format_list]
        self.not_equal_result(shape_format)

    def test_not_equal_shape_format_fp32_4d(self, device):
        format_list = [-1, 0, 3]
        shape_format = [[[np.float32, i, [32, 3, 3, 3]], [np.float32, i, [32, 3, 3, 3]]] for i in format_list]
        self.not_equal_result(shape_format)

    # scala-----------------------------------------------------------------

    def test_not_equal_scalar_shape_format_fp16_1d(self, device):
        format_list = [-1, 0, 3]
        shape_format = [[[np.float16, i, 18]] for i in format_list]
        self.not_equal_scalar_result(shape_format)

    def test_not_equal_scalar_shape_format_fp32_1d(self, device):
        format_list = [-1, 0, 3]
        shape_format = [[[np.float32, i, [18]]] for i in format_list]
        self.not_equal_scalar_result(shape_format)

    def test_not_equal_scalar_shape_format_fp16_2d(self, device):
        format_list = [-1, 0]
        shape_format = [[[np.float16, i, [64, 7]]] for i in format_list]
        self.not_equal_scalar_result(shape_format)

    def test_not_equal_scalar_shape_format_fp32_2d(self, device):
        format_list = [-1, 0]
        shape_format = [[[np.float32, i, [64, 7]]] for i in format_list]
        self.not_equal_scalar_result(shape_format)

    def test_not_equal_scalar_shape_format_fp32_3d(self, device):
        format_list = [-1, 0]
        shape_format = [[[np.float32, i, [64, 24, 38]]] for i in format_list]
        self.not_equal_scalar_result(shape_format)

    def test_not_equal_scalar_shape_format_fp16_4d(self, device):
        format_list = [-1, 0]
        shape_format = [[[np.float16, i, [32, 3, 3, 3]]] for i in format_list]
        self.not_equal_scalar_result(shape_format)

    def test_not_equal_scalar_shape_format_fp32_4d(self, device):
        format_list = [-1, 0]
        shape_format = [[[np.float32, i, [32, 3, 3, 3]]] for i in format_list]
        self.not_equal_scalar_result(shape_format)

    def test_not_equal_shape_format_int32_1d(self, device):
        format_list = [-1, 0]
        shape_format = [[[np.int32, i, [16]], [np.int32, i, [16]]] for i in format_list]
        self.not_equal_result(shape_format)

    def test_not_equal_shape_format_int32_2d(self, device):
        format_list = [-1, 0]
        shape_format = [[[np.int32, i, [448, 1]], [np.int32, i, [448, 1]]] for i in format_list]
        self.not_equal_result(shape_format)

    def test_not_equal_shape_format_int32_3d(self, device):
        format_list = [-1, 0]
        shape_format = [[[np.int32, i, [16, 640, 640]], [np.int32, i, [16, 640, 640]]] for i in format_list]
        self.not_equal_result(shape_format)

    def test_not_equal_shape_format_int32_4d(self, device):
        format_list = [-1, 0]
        shape_format = [[[np.int32, i, [32, 3, 3, 3]], [np.int32, i, [32, 3, 3, 3]]] for i in format_list]
        self.not_equal_result(shape_format)


instantiate_device_type_tests(TestNotEqual, globals(), except_for="cpu")

if __name__ == "__main__":
    run_tests()
