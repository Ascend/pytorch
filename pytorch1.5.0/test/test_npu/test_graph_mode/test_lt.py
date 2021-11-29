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
from graph_utils import RunFuncInGraphMode


class TestLt(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = torch.lt(input1, input2)
        output = output.numpy().astype(np.int32)
        return output

    def cpu_op_exec_out(self, input1, input2, input3):
        torch.lt(input1, input2, out = input3)
        output = input3.numpy().astype(np.int32)
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.lt(input1, input2)
        output = output.to("cpu")
        output = output.numpy().astype(np.int32)
        return output

    def cpu_op_inplace_exec(self, input1, input2):
        output = input1.lt_(input2)
        output = input1.numpy().astype(np.int32)
        return output

    def npu_op_inplace_exec(self, input1, input2):
        output = input1.lt_(input2)
        output = output.to("cpu")
        output = output.numpy().astype(np.int32)
        return output

    def npu_op_exec_out(self, input1, input2, out):
        torch.lt(input1, input2, out=out)
        output = out.to("cpu")
        output = output.numpy().astype(np.int32)
        return output

    def cpu_op_exec_scalar(self, input, scalar):
        output = torch.lt(input, scalar)
        output = output.numpy().astype(np.int32)
        return output

    def cpu_op_exec_scalar_out(self, input1, scalar, input2):
        torch.lt(input1, scalar, out = input2)
        output = input2.numpy().astype(np.int32)
        return output

    def npu_op_exec_scalar(self, input, scalar):
        output = torch.lt(input, scalar)
        output = output.to("cpu")
        output = output.numpy().astype(np.int32)
        return output

    def npu_op_exec_scalar_out(self, input, scalar, out):
        torch.lt(input, scalar, out=out)
        output = out.to("cpu")
        output = output.numpy().astype(np.int32)
        return output

    def lt_tensor_out_result(self, shape_format):
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

    @RunFuncInGraphMode
    def test_lt_tensor_out(self, device):
        shape_format = [
            [[np.float16, 0, [128, 116, 14, 14]], [np.float16, 0, [256, 116, 1, 1]]],
            [[np.float16, 0, [128, 3, 224, 224]], [np.float16, 0, [3, 3, 3]]],
            [[np.float16, 0, [128, 116, 14, 14]], [np.float16, 0, [128, 116, 14, 14]]],
            [[np.float32, 0, [256, 128, 7, 7]],   [np.float32, 0, [128, 256, 3, 3]]],
            [[np.float32, 0, [2, 3, 3, 3]],       [np.float32, 0, [3, 1, 3]]],
            [[np.float32, 0, [128, 232, 7, 7]],   [np.float32, 0, [128, 232, 7, 7]]],
        ]
        self.lt_tensor_out_result(shape_format)

    def lt_scalar_out_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_input2 = torch.randn(item[1][2]) < 0
            npu_input2 = cpu_input2.npu()
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            if cpu_input2.dtype == torch.float16:
                cpu_input2 = cpu_input2.to(torch.float32)
            scalar = np.random.uniform(0, 100)
            cpu_output_out = self.cpu_op_exec_scalar_out(cpu_input1, scalar, cpu_input2)
            npu_output_out = self.npu_op_exec_scalar_out(npu_input1, scalar, npu_input2)
            cpu_output_out = cpu_output_out.astype(npu_output_out.dtype)
            self.assertRtolEqual(cpu_output_out, npu_output_out)

    @RunFuncInGraphMode
    def test_lt_scalar_out(self, device):
        shape_format = [
            [[np.float16, 0, [4, 4, 128, 128]], [np.float16, 0, [256, 116, 1, 1]]],
            [[np.float16, 0, [12, 10, 14, 14]], [np.float16, 0, [256, 116, 1, 1]]],
            [[np.float16, 0, [16, 3, 1111, 1212]], [np.float16, 0, [3, 3, 3]]],
            [[np.float16, 0, [16, 16, 14, 14]], [np.float16, 0, [128, 116, 14, 14]]],
            [[np.float32, 0, [20, 10, 7, 7]], [np.float32, 0, [128, 256, 3, 3]]],
            [[np.float32, 0, [1313, 3, 3, 3]], [np.float32, 0, [3, 1, 3]]],
            [[np.float32, 0, [16, 22, 7, 7]], [np.float32, 0, [128, 232, 7, 7]]],
        ]
        self.lt_scalar_out_result(shape_format)

    def lt_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            cpu_output_inp = self.cpu_op_inplace_exec(cpu_input1, cpu_input2)
            npu_output_inp = self.npu_op_inplace_exec(npu_input1, npu_input2)

            cpu_output = cpu_output.astype(npu_output.dtype)
            cpu_output_inp = cpu_output_inp.astype(npu_output_inp.dtype)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output_inp, npu_output_inp)

    def lt_scalar_result(self, shape_format):
        for item in shape_format:
            scalar = np.random.uniform(0, 100)
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output_scalar = self.cpu_op_exec_scalar(cpu_input1, scalar)
            npu_output_scalar = self.npu_op_exec_scalar(npu_input1, scalar)

            cpu_output_scalar = cpu_output_scalar.astype(npu_output_scalar.dtype)

            self.assertRtolEqual(cpu_output_scalar, npu_output_scalar)

    @RunFuncInGraphMode
    def test_lt_shape_format_fp16_1d(self, device):
        format_list = [-1, 0]
        shape_format = [[np.float16, i, 5] for i in format_list]
        self.lt_result(shape_format)

    @RunFuncInGraphMode
    def test_lt_shape_format_fp32_1d(self, device):
        format_list = [-1, 0, 3]
        shape_format = [[np.float32, i, 5] for i in format_list]
        self.lt_result(shape_format)

    @RunFuncInGraphMode
    def test_lt_shape_format_fp16_2d(self, device):
        format_list = [-1, 0]
        shape_format = [[np.float16, i, [5, 3]] for i in format_list]
        self.lt_result(shape_format)

    @RunFuncInGraphMode
    def test_lt_shape_format_fp32_2d(self, device):
        format_list = [-1, 0]
        shape_format = [[np.float32, i, [5, 3]] for i in format_list]
        self.lt_result(shape_format)

    @RunFuncInGraphMode
    def test_lt_shape_format_fp16_3d(self, device):
        format_list = [-1, 0]
        shape_format = [[np.float16, i, [16, 640, 640]] for i in format_list]
        self.lt_result(shape_format)

    @RunFuncInGraphMode
    def test_lt_shape_format_fp32_3d(self, device):
        format_list = [-1, 0, 3]
        shape_format = [[np.float32, i, [16, 640, 640]] for i in format_list]
        self.lt_result(shape_format)

    @RunFuncInGraphMode
    def test_lt_shape_format_fp16_4d(self, device):
        format_list = [-1, 3]
        shape_format = [[np.float16, i, [32, 3, 3, 3]] for i in format_list]
        self.lt_result(shape_format)

    @RunFuncInGraphMode
    def test_lt_shape_format_fp32_4d(self, device):
        format_list = [-1, 3]
        shape_format = [[np.float32, i, [32, 3, 3, 3]] for i in format_list]
        self.lt_result(shape_format)

    # scalar-----------------------------------------------------------------------
    @RunFuncInGraphMode
    def test_lt_scalar_shape_format_fp16_1d(self, device):
        format_list = [-1, 0]
        shape_format = [[np.float16, i, 18] for i in format_list]
        self.lt_scalar_result(shape_format)

    @RunFuncInGraphMode
    def test_lt_scalar_shape_format_fp32_1d(self, device):
        format_list = [-1, 0]
        shape_format = [[np.float32, i, [18]] for i in format_list]
        self.lt_scalar_result(shape_format)

    @RunFuncInGraphMode
    def test_lt_scalar_shape_format_fp16_2d(self, device):
        format_list = [-1, 0]
        shape_format = [[np.float16, i, [5, 8]] for i in format_list]
        self.lt_scalar_result(shape_format)

    @RunFuncInGraphMode
    def test_lt_scalar_shape_format_fp32_2d(self, device):
        format_list = [-1, 0]
        shape_format = [[np.float32, i, [5, 8]] for i in format_list]
        self.lt_scalar_result(shape_format)

    @RunFuncInGraphMode
    def test_lt_scalar_shape_format_fp16_3d(self, device):
        format_list = [-1, 0]
        shape_format = [[np.float16, i, [4, 16, 32]] for i in format_list]
        self.lt_scalar_result(shape_format)

    @RunFuncInGraphMode
    def test_lt_scalar_shape_format_fp32_3d(self, device):
        format_list = [-1, 0]
        shape_format = [[np.float32, i, [4, 16, 32]] for i in format_list]
        self.lt_scalar_result(shape_format)

    @RunFuncInGraphMode
    def test_lt_scalar_shape_format_fp16_4d(self, device):
        format_list = [-1, 0]
        shape_format = [[np.float16, i, [32, 3, 3, 3]] for i in format_list]
        self.lt_scalar_result(shape_format)

    @RunFuncInGraphMode
    def test_lt_scalar_shape_format_fp32_4d(self, device):
        format_list = [-1, 0]
        shape_format = [[np.float32, i, [32, 3, 3, 3]] for i in format_list]
        self.lt_scalar_result(shape_format)

    @RunFuncInGraphMode
    def test_lt_mix_dtype(self, device):
        npu_input1, npu_input2 = create_common_tensor([np.float16, 0, (2, 3)], 1, 100)
        npu_input3, npu_input4 = create_common_tensor([np.float32, 0, (2, 3)], 1, 100)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input3)
        npu_output = self.npu_op_exec(npu_input2, npu_input4)
        self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestLt, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
