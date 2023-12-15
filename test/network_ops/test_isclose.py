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


class TestIsclose(TestCase):

    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
        return npu_input1, npu_input2

    def generate_nan(self, shape, dtype):
        input1 = np.full(shape, np.nan).astype(dtype)
        input2 = np.full(shape, np.nan).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
        return npu_input1, npu_input2

    def cpu_op_exec(self, input1, input2):
        output = torch.isclose(input1, input2)
        output = output.numpy()
        return output

    def cpu_op_exec_rtol_atol(self, input1, input2, rtol, atol):
        output = torch.isclose(input1, input2, rtol=rtol, atol=atol)
        output = output.numpy()
        return output

    def cpu_op_exec_equal_nan(self, input1, input2, equal_nan):
        output = torch.isclose(input1, input2, equal_nan=equal_nan)
        output = output.numpy()
        return output

    def npu_op_exec_tensor_need_to_npu(self, input1, input2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        output = torch.isclose(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_tensor_need_to_npu_rtol_atol(self, input1, input2, rtol, atol):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        output = torch.isclose(input1, input2, rtol=rtol, atol=atol)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_tensor_need_to_npu_equal_nan(self, input1, input2, equal_nan):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        output = torch.isclose(input1, input2, equal_nan=equal_nan)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_isclose_int32_float32(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(0, 100, (4, 3), np.int32)
        npu_input1 = npu_input1.to(torch.float32)
        npu_input2 = npu_input2.to(torch.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec_tensor_need_to_npu(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_isclose_equal_nan_false(self, device="npu"):
        npu_input1, npu_input2 = self.generate_nan((4, 3), np.int32)
        cpu_output = self.cpu_op_exec_equal_nan(npu_input1, npu_input2, False)
        npu_output = self.npu_op_exec_tensor_need_to_npu_equal_nan(npu_input1, npu_input2, False)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_isclose_equal_nan_true(self, device="npu"):
        npu_input1, npu_input2 = self.generate_nan((4, 3), np.int32)
        cpu_output = self.cpu_op_exec_equal_nan(npu_input1, npu_input2, True)
        npu_output = self.npu_op_exec_tensor_need_to_npu_equal_nan(npu_input1, npu_input2, True)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_isclose_int32_001(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(0, 100, (4, 3), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec_tensor_need_to_npu(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_isclose_int32_002(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(100, 100, (4, 3, 2), np.int32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec_tensor_need_to_npu(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_isclose_int32_003(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(0, 100, (4, 3, 2), np.int32)
        rtol = 8e-05
        atol = 8e-08
        cpu_output = self.cpu_op_exec_rtol_atol(npu_input1, npu_input2, rtol, atol)
        npu_output = self.npu_op_exec_tensor_need_to_npu_rtol_atol(npu_input1, npu_input2, rtol, atol)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_isclose_float32_001(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(100, 100, (4, 3), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec_tensor_need_to_npu(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_isclose_float32_002(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(0, 100, (4, 3, 2), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2)
        npu_output = self.npu_op_exec_tensor_need_to_npu(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_isclose_float32_003(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(0, 100, (4, 3, 2), np.float32)
        rtol = 8e-05
        atol = 8e-08
        cpu_output = self.cpu_op_exec_rtol_atol(npu_input1, npu_input2, rtol, atol)
        npu_output = self.npu_op_exec_tensor_need_to_npu_rtol_atol(npu_input1, npu_input2, rtol, atol)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_isclose_float16_001(self, device="npu"):
        def cpu_op_exec_fp16(input1, input2):
            input1 = input1.to(torch.float32)
            input2 = input2.to(torch.float32)
            output = torch.isclose(input1, input2)
            output = output.numpy()
            return output
        npu_input1, npu_input2 = self.generate_data(0, 100, (5, 3), np.float16)
        cpu_output = cpu_op_exec_fp16(npu_input1, npu_input2)
        npu_output = self.npu_op_exec_tensor_need_to_npu(npu_input1, npu_input2)
        cpu_output = cpu_output.astype(npu_output.dtype)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_isclose_float16_002(self, device="npu"):
        def cpu_op_exec_fp16(input1, input2):
            input1 = input1.to(torch.float32)
            input2 = input2.to(torch.float32)
            output = torch.isclose(input1, input2)
            output = output.numpy()
            return output
        npu_input1, npu_input2 = self.generate_data(100, 100, (5, 3, 2), np.float16)
        cpu_output = cpu_op_exec_fp16(npu_input1, npu_input2)
        npu_output = self.npu_op_exec_tensor_need_to_npu(npu_input1, npu_input2)
        cpu_output = cpu_output.astype(npu_output.dtype)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_isclose_float16_003(self, device="npu"):
        def cpu_op_exec_fp16_rtol_atol(input1, input2, rtol, atol):
            input1 = input1.to(torch.float32)
            input2 = input2.to(torch.float32)
            output = torch.isclose(input1, input2, rtol=rtol, atol=atol)
            output = output.numpy()
            return output
        npu_input1, npu_input2 = self.generate_data(0, 100, (4, 3, 2), np.float16)
        rtol = 8e-05
        atol = 8e-08
        cpu_output = cpu_op_exec_fp16_rtol_atol(npu_input1, npu_input2, rtol, atol)
        npu_output = self.npu_op_exec_tensor_need_to_npu_rtol_atol(npu_input1, npu_input2, rtol, atol)
        cpu_output = cpu_output.astype(npu_output.dtype)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
