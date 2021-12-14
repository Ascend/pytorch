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
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestAddcdiv(TestCase):
    def test_addcdiv(self, device):
        def _test_addcdiv(a, alpha, b, c):
            actual = torch.addcdiv(a, b, c, value=alpha)
            # implementation of addcdiv downcasts alpha. arithmetic ops don't.
            if not actual.dtype.is_floating_point:
                alpha = int(alpha)
            expected = a + (alpha * b) / c
            # print(expected)
            # print(actual)
            self.assertTrue(torch.allclose(expected.to("cpu"), actual.to("cpu"), equal_nan=True))

            with self.maybeWarnsRegex(
                    UserWarning, "This overload of addcdiv is deprecated"):
                self.assertEqual(actual.to("cpu"), torch.addcdiv(a, alpha, b, c).to("cpu"))

        def non_zero_rand(size, dtype, device):
            if dtype.is_floating_point:
                a = torch.rand(size=size, dtype=dtype, device="cpu")
                a = a.to("npu")  # torch.rand()在npu暂未适配
            elif dtype == torch.uint8:
                a = torch.randint(1, 5, size=size, dtype=dtype, device=device)
            else:
                a = torch.randint(-5, 5, size=size, dtype=dtype, device=device)
            # return a + (a == 0).type(dtype)  #add 方法有些问题，先注释不使用
            return a.type(dtype)

        for dtype in torch.testing.get_all_math_dtypes(device):
            # print(dtype, " : ", device)
            if dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.float64]:
                continue
            _test_addcdiv(
                non_zero_rand((2, 2), dtype=dtype, device=device),
                0.5,
                non_zero_rand((2, 2), dtype=dtype, device=device),
                non_zero_rand((2, 2), dtype=dtype, device=device))

    def generate_data(self, min, max, shape, dtype):
        input1 = np.random.uniform(min, max, shape).astype(dtype)
        input2 = np.random.uniform(min, max, shape).astype(dtype)
        input3 = np.random.uniform(min, max, shape).astype(dtype)

        # 将numpy.ndarray转换为torch.tensor
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
        npu_input3 = torch.from_numpy(input3)

        return npu_input1, npu_input2, npu_input3

    def generate_single_data(self, min, max, shape, dtype):
        input = np.random.uniform(min, max, shape).astype(dtype)
        npu_input = torch.from_numpy(input)
        return npu_input

    def generate_scalar(self, min, max):
        scalar = np.random.uniform(min, max)
        return scalar

    def generate_int_scalar(self, min, max):
        scalar = np.random.randint(min, max)
        return scalar

    def test_addcdiv_float32(self, device):
        def cpu_op_exec(input1, input2, input3, scalar):
            ori_dtype = input1.dtype
            if ori_dtype == torch.float16:
                input1 = input1.to(torch.float32)
                input2 = input2.to(torch.float32)
                input3 = input3.to(torch.float32)
            output = torch.addcdiv(input1, input2, input3, value=scalar)
            if ori_dtype == torch.float16:
                output = output.to(ori_dtype)
            return output

        def npu_op_exec(input1, input2, input3, scalar):
            input1 = input1.to("npu")
            input2 = input2.to("npu")
            input3 = input3.to("npu")
            output = torch.addcdiv(input1, input2, input3, value=scalar)
            output = output.to("cpu")
            return output
        dtype_list = [np.float32, np.float16]
        for dtype in dtype_list:
            npu_input1, npu_input2, npu_input3 = self.generate_data(1, 100, (5, 3), dtype)
            scalar = self.generate_scalar(1, 10)
            cpu_output = cpu_op_exec(npu_input1, npu_input2, npu_input3, scalar)
            npu_output = npu_op_exec(npu_input1, npu_input2, npu_input3, scalar)
            self.assertEqual(cpu_output, npu_output)


    def test_addcdiv_float32_out(self, device):
        def cpu_op_exec_out(input1, input2, input3, scalar, input4):
            ori_dtype = input1.dtype
            if ori_dtype == torch.float16:
                input1 = input1.to(torch.float32)
                input2 = input2.to(torch.float32)
                input3 = input3.to(torch.float32)
                input4 = input4.to(torch.float32)
            output = input4
            torch.addcdiv(input1, input2, input3, value=scalar, out=output)
            if ori_dtype == torch.float16:
                output = output.to(ori_dtype)
            output = output.numpy()
            return output

        def npu_op_exec_out(input1, input2, input3, scalar, input4):
            input1 = input1.to("npu")
            input2 = input2.to("npu")
            input3 = input3.to("npu")
            output = input4.to("npu")
            torch.addcdiv(input1, input2, input3, value=scalar, out=output)
            output = output.to("cpu")
            output = output.numpy()
            return output
        dtype_list = [np.float32, np.float16]
        for dtype in dtype_list:
            npu_input1, npu_input2, npu_input3 = self.generate_data(1, 100, (5, 3), dtype)
            scalar = self.generate_scalar(1, 10)
            npu_input4 = self.generate_single_data(1, 100, (5, 3), dtype)
            cpu_output = cpu_op_exec_out(npu_input1, npu_input2, npu_input3, scalar, npu_input4)
            npu_output = npu_op_exec_out(npu_input1, npu_input2, npu_input3, scalar, npu_input4)
            self.assertEqual(cpu_output, npu_output)

    def test_addcdiv_float32_broadcast(self, device):
        def cpu_op_exec(input1, input2, input3, scalar):
            ori_dtype = input1.dtype
            if ori_dtype == torch.float16:
                input1 = input1.to(torch.float32)
                input2 = input2.to(torch.float32)
                input3 = input3.to(torch.float32)
            output = torch.addcdiv(input1, input2, input3, value=scalar)
            if ori_dtype == torch.float16:
                output = output.to(ori_dtype)
            return output

        def npu_op_exec(input1, input2, input3, scalar):
            input1 = input1.to("npu")
            input2 = input2.to("npu")
            input3 = input3.to("npu")
            output = torch.addcdiv(input1, input2, input3, value=scalar)
            output = output.to("cpu")
            return output
        dtype_list = [np.float32, np.float16]
        for dtype in dtype_list:
            npu_input1 = self.generate_single_data(1, 100, (5, 3, 1), dtype)
            npu_input2 = self.generate_single_data(1, 100, (5, 1, 5), dtype)
            npu_input3 = self.generate_single_data(1, 100, (1, 1, 5), dtype)
            scalar = self.generate_scalar(1, 10)
            cpu_output = cpu_op_exec(npu_input1, npu_input2, npu_input3, scalar)
            npu_output = npu_op_exec(npu_input1, npu_input2, npu_input3, scalar)
            # self.assertEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_addcdiv_inp_contiguous_float32(self, device):
        def cpu_op_inp_contiguous_exec(input1, input2, input3, scalar):
            ori_dtype = input1.dtype
            if ori_dtype == torch.float16:
                input1 = input1.to(torch.float32)
                input2 = input2.to(torch.float32)
                input3 = input3.to(torch.float32)
            input1.addcdiv_(input2, input3, value=scalar)
            if ori_dtype == torch.float16:
                input1 = input1.to(ori_dtype)
            output = input1.numpy()
            return output

        def npu_op_inp_contiguous_exec(input1, input2, input3, scalar):
            input1 = input1.to("npu")
            input2 = input2.to("npu")
            input3 = input3.to("npu")
            input1.addcdiv_(input2, input3, value=scalar)
            output = input1.to("cpu")
            output = output.numpy()
            return output
        dtype_list = [np.float32, np.float16]
        for dtype in dtype_list:
            npu_input1, npu_input2, npu_input3 = self.generate_data(1, 100, (5, 3), dtype)
            cpu_input1 = copy.deepcopy(npu_input1)
            cpu_input2 = copy.deepcopy(npu_input2)
            cpu_input3 = copy.deepcopy(npu_input3)
            scalar = self.generate_int_scalar(1, 10)
            cpu_output = cpu_op_inp_contiguous_exec(cpu_input1, cpu_input2, cpu_input3, scalar)
            npu_output = npu_op_inp_contiguous_exec(npu_input1, npu_input2, npu_input3, scalar)
            self.assertEqual(cpu_output, npu_output)

    def test_addcdiv_inp_input1_noncontiguous_float32(self, device):
        def cpu_op_inp_input1_noncontiguous_exec(input1, input2, input3, scalar):
            ori_dtype = input1.dtype
            if ori_dtype == torch.float16:
                input1 = input1.to(torch.float32)
                input2 = input2.to(torch.float32)
                input3 = input3.to(torch.float32)
            input1_strided = input1.as_strided([2, 2], [1, 2], 2)
            input1_strided.addcdiv_(input2, input3, value=scalar)
            if ori_dtype == torch.float16:
                input1 = input1.to(ori_dtype)
            output = input1.numpy()
            return output

        def npu_op_inp_input1_noncontiguous_exec(input1, input2, input3, scalar):
            input1 = input1.to("npu")
            input2 = input2.to("npu")
            input3 = input3.to("npu")
            input1_as_strided = input1.as_strided([2, 2], [1, 2], 2)
            input1_as_strided.addcdiv_(input2, input3, value=scalar)
            output = input1.to("cpu")
            output = output.numpy()
            return output
        dtype_list = [np.float32, np.float16]
        for dtype in dtype_list:
            npu_input1 = self.generate_single_data(1, 100, (4, 3), dtype)
            npu_input2 = self.generate_single_data(1, 100, (2, 2), dtype)
            npu_input3 = self.generate_single_data(1, 100, (2, 2), dtype)
            cpu_input1 = copy.deepcopy(npu_input1)
            cpu_input2 = copy.deepcopy(npu_input2)
            cpu_input3 = copy.deepcopy(npu_input3)
            scalar = self.generate_int_scalar(1, 10)
            cpu_output = cpu_op_inp_input1_noncontiguous_exec(cpu_input1, cpu_input2, cpu_input3, scalar)
            npu_output = npu_op_inp_input1_noncontiguous_exec(npu_input1, npu_input2, npu_input3, scalar)
            self.assertEqual(cpu_output, npu_output)

    def test_addcdiv_inp_input2_noncontiguous_float32(self, device):
        def cpu_op_inp_input2_noncontiguous_exec(input1, input2, input3, scalar):
            ori_dtype = input1.dtype
            if ori_dtype == torch.float16:
                input1 = input1.to(torch.float32)
                input2 = input2.to(torch.float32)
                input3 = input3.to(torch.float32)
            input2_strided = input2.as_strided([2, 2], [1, 2], 2)
            input1.addcdiv_(input2_strided, input3, value=scalar)
            if ori_dtype == torch.float16:
                input1 = input1.to(ori_dtype)
            output = input1.numpy()
            return output

        def npu_op_inp_input2_noncontiguous_exec(input1, input2, input3, scalar):
            input1 = input1.to("npu")
            input3 = input3.to("npu")
            input2 = input2.to("npu")
            input2_as_strided = input2.as_strided([2, 2], [1, 2], 2)
            input1.addcdiv_(input2_as_strided, input3, value=scalar)
            output = input1.to("cpu")
            output = output.numpy()
            return output

        dtype_list = [np.float32, np.float16]
        for dtype in dtype_list:
            npu_input1 = self.generate_single_data(1, 100, (2, 2), dtype)
            npu_input2 = self.generate_single_data(1, 100, (4, 3), dtype)
            npu_input3 = self.generate_single_data(1, 100, (2, 2), dtype)
            cpu_input1 = copy.deepcopy(npu_input1)
            cpu_input2 = copy.deepcopy(npu_input2)
            cpu_input3 = copy.deepcopy(npu_input3)
            scalar = self.generate_int_scalar(1, 10)
            cpu_output = cpu_op_inp_input2_noncontiguous_exec(cpu_input1, cpu_input2, cpu_input3, scalar)
            npu_output = npu_op_inp_input2_noncontiguous_exec(npu_input1, npu_input2, npu_input3, scalar)
            self.assertEqual(cpu_output, npu_output)

    def test_addcdiv_inp_input3_noncontiguous_fp32_fp16(self, device):
        def cpu_op_inp_input3_noncontiguous_exec(input1, input2, input3, scalar):
            ori_dtype = input1.dtype
            if ori_dtype == torch.float16:
                input1 = input1.to(torch.float32)
                input2 = input2.to(torch.float32)
                input3 = input3.to(torch.float32)
            input3_strided = input3.as_strided([2, 2], [1, 2], 2)
            input1.addcdiv_(input2, input3_strided, value=scalar)
            if ori_dtype == torch.float16:
                input1 = input1.to(ori_dtype)
            output = input1.numpy()
            return output

        def npu_op_inp_input3_noncontiguous_exec(input1, input2, input3, scalar):
            input1 = input1.to("npu")
            input2 = input2.to("npu")
            input3 = input3.to("npu")
            input3_as_strided = input3.as_strided([2, 2], [1, 2], 2)
            input1.addcdiv_(input2, input3_as_strided, value=scalar)
            output = input1.to("cpu")
            output = output.numpy()
            return output

        dtype_list = [np.float32, np.float16]
        for dtype in dtype_list:
            npu_input1 = self.generate_single_data(1, 100, (2, 2), dtype)
            npu_input2 = self.generate_single_data(1, 100, (2, 2), dtype)
            npu_input3 = self.generate_single_data(1, 100, (4, 3), dtype)
            cpu_input1 = copy.deepcopy(npu_input1)
            cpu_input2 = copy.deepcopy(npu_input2)
            cpu_input3 = copy.deepcopy(npu_input3)
            scalar = self.generate_int_scalar(1, 10)
            cpu_output = cpu_op_inp_input3_noncontiguous_exec(cpu_input1, cpu_input2, cpu_input3, scalar)
            npu_output = npu_op_inp_input3_noncontiguous_exec(npu_input1, npu_input2, npu_input3, scalar)
            self.assertEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestAddcdiv, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
