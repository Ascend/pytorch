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


import copy
import torch
import numpy as np

from torch_npu.testing.testcase import TestCase, run_tests


class TestAddcdiv(TestCase):

    def cpu_op_inp_input3_noncontiguous_exec(self, input1, input2, input3, scalar):
        input3_strided = input3.as_strided([2, 2], [1, 2], 2)
        input1.addcdiv_(input2, input3_strided, value=scalar)
        output = input1.numpy()
        return output

    def npu_op_inp_input3_noncontiguous_exec(self, input1, input2, input3, scalar):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        input3 = input3.to("npu")
        input3_as_strided = input3.as_strided([2, 2], [1, 2], 2)
        input1.addcdiv_(input2, input3_as_strided, value=scalar)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def non_zero_rand(self, size, dtype, device="npu"):
        if dtype.is_floating_point:
            a = torch.rand(size=size, dtype=dtype, device="cpu")
            a = a.to("npu")
        elif dtype == torch.uint8:
            a = torch.randint(1, 5, size=size, dtype=dtype, device="cpu").to(device)
        else:
            a = torch.randint(-5, 5, size=size, dtype=dtype, device="cpu").to(device)
        return a.type(dtype)

    def cpu_op_exec(self, input1, input2, input3, scalar):
        output = torch.addcdiv(input1, input2, input3, value=scalar)
        return output

    def npu_op_exec(self, input1, input2, input3, scalar):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        input3 = input3.to("npu")
        output = torch.addcdiv(input1, input2, input3, value=scalar)
        output = output.to("cpu")
        return output

    def cpu_op_exec_out(self, input1, input2, input3, scalar, output):
        torch.addcdiv(input1, input2, input3, value=scalar, out=output)
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2, input3, scalar, output):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        input3 = input3.to("npu")
        output = output.to("npu")
        torch.addcdiv(input1, input2, input3, value=scalar, out=output)
        output = output.to("cpu").numpy()
        return output

    def cpu_op_inp_contiguous_exec(self, input1, input2, input3, scalar):
        input1.addcdiv_(input2, input3, value=scalar)
        output = input1.numpy()
        return output

    def npu_op_inp_contiguous_exec(self, input1, input2, input3, scalar):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        input3 = input3.to("npu")
        input1.addcdiv_(input2, input3, value=scalar)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_inp_input1_noncontiguous_exec(self, input1, input2, input3, scalar):
        input1_strided = input1.as_strided([2, 2], [1, 2], 2)
        input1_strided.addcdiv_(input2, input3, value=scalar)
        output = input1.numpy()
        return output

    def npu_op_inp_input1_noncontiguous_exec(self, input1, input2, input3, scalar):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        input3 = input3.to("npu")
        input1_as_strided = input1.as_strided([2, 2], [1, 2], 2)
        input1_as_strided.addcdiv_(input2, input3, value=scalar)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_inp_input2_noncontiguous_exec(self, input1, input2, input3, scalar):
        input2_strided = input2.as_strided([2, 2], [1, 2], 2)
        input1.addcdiv_(input2_strided, input3, value=scalar)
        output = input1.numpy()
        return output

    def npu_op_inp_input2_noncontiguous_exec(self, input1, input2, input3, scalar):
        input1 = input1.to("npu")
        input3 = input3.to("npu")
        input2 = input2.to("npu")
        input2_as_strided = input2.as_strided([2, 2], [1, 2], 2)
        input1.addcdiv_(input2_as_strided, input3, value=scalar)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def generate_data(self, min1, max1, shape, dtype):
        input1 = np.random.uniform(min1, max1, shape).astype(dtype)
        input2 = np.random.uniform(min1, max1, shape).astype(dtype)
        input3 = np.random.uniform(min1, max1, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
        npu_input3 = torch.from_numpy(input3)
        return npu_input1, npu_input2, npu_input3

    def generate_single_data(self, min1, max1, shape, dtype):
        inputs = np.random.uniform(min1, max1, shape).astype(dtype)
        npu_input = torch.from_numpy(inputs)
        return npu_input

    def generate_scalar(self, min1, max1):
        scalar = np.random.uniform(min1, max1)
        return scalar

    def generate_int_scalar(self, min1, max1):
        scalar = np.random.randint(min1, max1)
        return scalar

    def _test_addcdiv(self, a, alpha, b, c):
        actual = torch.addcdiv(a, b, c, value=alpha)
        if not actual.dtype.is_floating_point:
            alpha = int(alpha)
        expected = a + (alpha * b) / c
        self.assertTrue(torch.allclose(expected.to("cpu"), actual.to("cpu"), equal_nan=True))
        self.assertRtolEqual(actual.to("cpu"), torch.addcdiv(a, alpha, b, c).to("cpu"))

    def test_addcdiv(self, device="npu"):
        """NPU does not support numpy.bool.

        with self.maybeWarnsRegex(UserWarning, "This overload of addcdiv is deprecated"):
            self.assertRtolEqual(actual.to("cpu"), torch.addcdiv(a, alpha, b, c).to("cpu"))

        """
        dtype_list = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
                      torch.float64, torch.complex64, torch.complex128]
        for dtype in torch.testing.get_all_math_dtypes(device):
            if dtype in dtype_list:
                continue
            self._test_addcdiv(
                self.non_zero_rand((2, 2), dtype=dtype, device=device),
                0.5,
                self.non_zero_rand((2, 2), dtype=dtype, device=device),
                self.non_zero_rand((2, 2), dtype=dtype, device=device))

    def test_addcdiv_float32(self):
        npu_input1, npu_input2, npu_input3 = self.generate_data(1, 100, (5, 3), np.float32)
        scalar = self.generate_scalar(1, 10)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, npu_input3, scalar)
        npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_input3, scalar)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_addcdiv_float32_out(self):
        npu_input1, npu_input2, npu_input3 = self.generate_data(1, 100, (5, 3), np.float32)
        scalar = self.generate_scalar(1, 10)
        npu_input4 = self.generate_single_data(1, 100, (5, 3), np.float32)
        cpu_output = self.cpu_op_exec_out(npu_input1, npu_input2, npu_input3, scalar, npu_input4)
        npu_output = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3, scalar, npu_input4)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_addcdiv_float32_broadcast(self):
        npu_input1 = self.generate_single_data(1, 100, (5, 3, 1), np.float32)
        npu_input2 = self.generate_single_data(1, 100, (5, 1, 5), np.float32)
        npu_input3 = self.generate_single_data(1, 100, (1, 1, 5), np.float32)
        scalar = self.generate_scalar(1, 10)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2, npu_input3, scalar)
        npu_output = self.npu_op_exec(npu_input1, npu_input2, npu_input3, scalar)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_addcdiv_inp_contiguous_float32(self):
        npu_input1, npu_input2, npu_input3 = self.generate_data(1, 100, (5, 3), np.float32)
        cpu_input1 = copy.deepcopy(npu_input1)
        cpu_input2 = copy.deepcopy(npu_input2)
        cpu_input3 = copy.deepcopy(npu_input3)
        scalar = self.generate_int_scalar(1, 10)
        cpu_output = self.cpu_op_inp_contiguous_exec(cpu_input1, cpu_input2, cpu_input3, scalar)
        npu_output = self.npu_op_inp_contiguous_exec(npu_input1, npu_input2, npu_input3, scalar)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_addcdiv_inp_input1_noncontiguous_float32(self):
        npu_input1 = self.generate_single_data(1, 100, (4, 3), np.float32)
        npu_input2 = self.generate_single_data(1, 100, (2, 2), np.float32)
        npu_input3 = self.generate_single_data(1, 100, (2, 2), np.float32)
        cpu_input1 = copy.deepcopy(npu_input1)
        cpu_input2 = copy.deepcopy(npu_input2)
        cpu_input3 = copy.deepcopy(npu_input3)
        scalar = self.generate_int_scalar(1, 10)
        cpu_output = self.cpu_op_inp_input1_noncontiguous_exec(cpu_input1, cpu_input2, cpu_input3, scalar)
        npu_output = self.npu_op_inp_input1_noncontiguous_exec(npu_input1, npu_input2, npu_input3, scalar)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_addcdiv_inp_input2_noncontiguous_float32(self):
        npu_input1 = self.generate_single_data(1, 100, (2, 2), np.float32)
        npu_input2 = self.generate_single_data(1, 100, (4, 3), np.float32)
        npu_input3 = self.generate_single_data(1, 100, (2, 2), np.float32)
        cpu_input1 = copy.deepcopy(npu_input1)
        cpu_input2 = copy.deepcopy(npu_input2)
        cpu_input3 = copy.deepcopy(npu_input3)
        scalar = self.generate_int_scalar(1, 10)
        cpu_output = self.cpu_op_inp_input2_noncontiguous_exec(cpu_input1, cpu_input2, cpu_input3, scalar)
        npu_output = self.npu_op_inp_input2_noncontiguous_exec(npu_input1, npu_input2, npu_input3, scalar)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_addcdiv_inp_input3_noncontiguous_float32(self):
        npu_input1 = self.generate_single_data(1, 100, (2, 2), np.float32)
        npu_input2 = self.generate_single_data(1, 100, (2, 2), np.float32)
        npu_input3 = self.generate_single_data(1, 100, (4, 3), np.float32)
        cpu_input1 = copy.deepcopy(npu_input1)
        cpu_input2 = copy.deepcopy(npu_input2)
        cpu_input3 = copy.deepcopy(npu_input3)
        scalar = self.generate_int_scalar(1, 10)
        cpu_output = self.cpu_op_inp_input3_noncontiguous_exec(cpu_input1, cpu_input2, cpu_input3, scalar)
        npu_output = self.npu_op_inp_input3_noncontiguous_exec(npu_input1, npu_input2, npu_input3, scalar)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_addcdiv_float64(self):
        cpu_input1, cpu_input2, cpu_input3 = self.generate_data(1, 100, (5, 3), np.float64)
        scalar = self.generate_scalar(1, 10)
        cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, cpu_input3, scalar)
        npu_output = self.npu_op_exec(cpu_input1, cpu_input2, cpu_input3, scalar)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_addcdiv_float16(self):
        cpu_input1, cpu_input2, cpu_input3 = self.generate_data(1, 100, (5, 3), np.float16)
        scalar = self.generate_scalar(1, 10)
        cpu_output = self.cpu_op_exec(cpu_input1.float(), cpu_input2.float(), cpu_input3.float(), scalar)
        npu_output = self.npu_op_exec(cpu_input1, cpu_input2, cpu_input3, scalar)
        cpu_output = cpu_output.to(npu_output.dtype)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
