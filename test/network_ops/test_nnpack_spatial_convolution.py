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

import unittest
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestNnpackSpatialConvolution(TestCase):

    def generate_data(self, min_d, max_d, N, C0, Hi, Wi, C1, Hw, Ww, dtype):
        input_shape = (N, C0, Hi, Wi)
        input_x = np.random.uniform(min_d, max_d, input_shape).astype(np.float16).astype(dtype)
        weight_shape = (C1, C0, Hw, Ww)
        weight = np.random.uniform(min_d, max_d, weight_shape).astype(np.float16).astype(dtype)
        input_x = torch.from_numpy(input_x)
        weight = torch.from_numpy(weight)
        bias = np.zeros(C1).astype(np.float16).astype(dtype)
        bias = torch.from_numpy(bias)
        padding = tuple(np.ones(2).astype(np.int))
        list1 = [input_x, weight, bias, padding]
        return list1

    @unittest.skipIf(not torch._nnpack_available(), "NNPACK unavailable")
    def cpu_op_exec(self, input_x, weight, bias, padding):
        flag = 0
        if input_x.dtype == torch.float16:
            input_x = input_x.to(torch.float32)
            weight = weight.to(torch.float32)
            bias = bias.to(torch.float32)
            flag = 1
        output = torch._nnpack_spatial_convolution(
            input_x, weight, bias, padding)
        if flag == 1:
            output = output.to(torch.float16)
        output = output.numpy()
        return output

    @unittest.skipIf(not torch._nnpack_available(), "NNPACK unavailable")
    def npu_op_exec(self, input_x, weight, bias, padding):
        flag = 0
        if input_x.dtype == torch.float16:
            input_x = input_x.to(torch.float32)
            weight = weight.to(torch.float32)
            bias = bias.to(torch.float32)
            flag = 1
        input_x = input_x.to("npu")
        weight = weight.to("npu")
        bias = bias.to("npu")
        output = torch._nnpack_spatial_convolution(
            input_x, weight, bias, padding)
        output = output.to("cpu")
        if flag == 1:
            output = output.to(torch.float16)
        output = output.numpy()
        return output

    def test__nnpack_spatial_convolution_float16_1(self, device="npu"):
        getlist1 = self.generate_data(
            -2, 2, 1, 3, 4, 4, 2, 2, 2, np.float16)
        cpu_output = self.cpu_op_exec(getlist1[0], getlist1[1], getlist1[2], getlist1[3])
        npu_output = self.npu_op_exec(getlist1[0], getlist1[1], getlist1[2], getlist1[3])
        self.assertRtolEqual(cpu_output, npu_output)

    def test__nnpack_spatial_convolution_float16_2(self, device="npu"):
        getlist1 = self.generate_data(
            -50, 50, 1, 3, 5, 5, 5, 2, 2, np.float16)
        cpu_output = self.cpu_op_exec(getlist1[0], getlist1[1], getlist1[2], getlist1[3])
        npu_output = self.npu_op_exec(getlist1[0], getlist1[1], getlist1[2], getlist1[3])
        self.assertRtolEqual(cpu_output, npu_output)

    def test__nnpack_spatial_convolution_float16_3(self, device="npu"):
        getlist1 = self.generate_data(
            -50, 50, 1, 5, 1024, 1024, 5, 8, 8, np.float16)
        cpu_output = self.cpu_op_exec(getlist1[0], getlist1[1], getlist1[2], getlist1[3])
        npu_output = self.npu_op_exec(getlist1[0], getlist1[1], getlist1[2], getlist1[3])
        self.assertRtolEqual(cpu_output, npu_output)

    def test__nnpack_spatial_convolution_float32_1(self, device="npu"):
        getlist1 = self.generate_data(
            -2, 2, 1, 3, 4, 4, 2, 2, 2, np.float32)
        cpu_output = self.cpu_op_exec(getlist1[0], getlist1[1], getlist1[2], getlist1[3])
        npu_output = self.npu_op_exec(getlist1[0], getlist1[1], getlist1[2], getlist1[3])
        self.assertRtolEqual(cpu_output, npu_output)

    def test__nnpack_spatial_convolution_float32_2(self, device="npu"):
        getlist1 = self.generate_data(
            -50, 50, 1, 3, 4, 4, 2, 2, 2, np.float32)
        cpu_output = self.cpu_op_exec(getlist1[0], getlist1[1], getlist1[2], getlist1[3])
        npu_output = self.npu_op_exec(getlist1[0], getlist1[1], getlist1[2], getlist1[3])
        self.assertRtolEqual(cpu_output, npu_output)

    def test__nnpack_spatial_convolution_float32_3(self, device="npu"):
        getlist1 = self.generate_data(
            -50, 50, 1, 5, 512, 512, 5, 8, 8, np.float32)
        cpu_output = self.cpu_op_exec(getlist1[0], getlist1[1], getlist1[2], getlist1[3])
        npu_output = self.npu_op_exec(getlist1[0], getlist1[1], getlist1[2], getlist1[3])
        self.assertRtolEqual(cpu_output, npu_output, prec=1e-3)


if __name__ == "__main__":
    run_tests()
