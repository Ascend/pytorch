# Copyright (c) 2020, Huawei Technologies.
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


class TestLeakRelu(TestCase):
    def cpu_op_exec(self, input1):
        m = torch.nn.LeakyReLU(0.01)
        output = m(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        m = torch.nn.LeakyReLU(0.01).to("npu")
        output = m(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_negative_slope_exec(self, input1, negativeSlope):
        output = torch.nn.functional.leaky_relu(input1, negative_slope=negativeSlope)
        output = output.numpy()
        return output

    def npu_op_negative_slope_exec(self, input1, negativeSlope):
        output = torch.nn.functional.leaky_relu(input1, negative_slope=negativeSlope)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def create_shape_format16(self, device="npu"):
        dtype_list = [np.float16]
        shape_list = [(1, 6, 4), (2, 4, 5)]
        format_list = [0, 3]
        negative_slope_list = [0.05, 0.1]

        shape_format = [[[i, j, k], h]
                        for i in dtype_list for j in format_list for k in shape_list for h in negative_slope_list]

        return shape_format

    def create_shape_format32(self, device="npu"):
        dtype_list1 = [np.float32]
        shape_list1 = [(1, 6, 4), (1, 4, 8), (1, 6, 8),
                       (2, 4, 5), (2, 5, 10), (2, 4, 10)]
        format_list1 = [0, 3]
        negative_slope_list1 = [0.02, 0.03]

        shape_format1 = [[[i, j, k], h]
                         for i in dtype_list1 for j in format_list1 for k in shape_list1 for h in negative_slope_list1]

        return shape_format1

    def test_leaky_relu_shape_format(self, device="npu"):
        for item in self.create_shape_format32(device):
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_leaky_relu_shape_format_fp16(self, device="npu"):
        def cpu_op_exec_fp16(input1):
            input1 = input1.to(torch.float32)
            m = torch.nn.LeakyReLU(0.01)
            output = m(input1)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        for item in self.create_shape_format16(device):
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_output = cpu_op_exec_fp16(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_leaky_relu_negative_slope_shape_format(self, device="npu"):
        for item in self.create_shape_format32(device):
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_negative_slope_exec(cpu_input, item[1])
            npu_output = self.npu_op_negative_slope_exec(npu_input, item[1])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_leaky_relu_negative_slope_shape_format_fp16(self, device="npu"):
        def cpu_op_negative_slope_exec_fp16(input1, negativeSlope):
            input1 = input1.to(torch.float32)
            output = torch.nn.functional.leaky_relu(input1, negative_slope=negativeSlope)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        for item in self.create_shape_format16(device):
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_output = cpu_op_negative_slope_exec_fp16(cpu_input, item[1])
            npu_output = self.npu_op_negative_slope_exec(npu_input, item[1])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
