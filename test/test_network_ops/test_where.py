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


class TestWhere(TestCase):
    def cpu_op_exec(self, input1):
        output = torch.where(input1)
        output = list(output)

        for i, _ in enumerate(output):
            output[i] = output[i].numpy().astype(np.int32)
        return output

    def npu_op_exec(self, input1):
        output = torch.where(input1)
        output = list(output)

        for i, _ in enumerate(output):
            output[i] = output[i].to("cpu").numpy().astype(np.int32)
        return output

    def cpu_op_exec_condition(self, input1, ones):
        output = torch.where(input1 > 0, input1, ones)
        output = output.numpy()
        return output

    def npu_op_exec_condition(self, input1, ones):
        output = torch.where(input1 > 0, input1, ones)
        output = output.to("cpu").numpy()
        return output

    def cpu_op_exec_s(self, input1, ones):
        output = torch._s_where(input1 > 0, input1, ones)
        output = output.numpy()
        return output

    def npu_op_exec_s(self, input1, ones):
        output = torch._s_where(input1 > 0, input1, ones)
        output = output.to("cpu").numpy()
        return output

    def where_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, -100, 100)
            cpu_ones = torch.ones_like(cpu_input1)
            npu_ones = cpu_ones.to("npu")
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_ones = cpu_ones.to(torch.float32)

            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)

            cpu_output_cond = self.cpu_op_exec_condition(cpu_input1, cpu_ones)
            npu_output_cond = self.npu_op_exec_condition(npu_input1, npu_ones)
            cpu_output_cond = cpu_output_cond.astype(npu_output_cond.dtype)

            cpu_output_s = self.cpu_op_exec_s(cpu_input1, cpu_ones)
            npu_output_s = self.npu_op_exec_s(npu_input1, npu_ones)
            cpu_output_s = cpu_output_s.astype(npu_output_s.dtype)

            for i, _ in enumerate(cpu_output):
                cpu_output[i] = cpu_output[i].astype(npu_output[i].dtype)
                self.assertRtolEqual(cpu_output[i], npu_output[i])

            self.assertRtolEqual(cpu_output_cond, npu_output_cond)
            self.assertRtolEqual(cpu_output_s, npu_output_s)

    def test_where_shape_format_fp32_1d(self, device="npu"):
        format_list = [0, 3]
        shape_format = [[np.float32, i, [18]] for i in format_list]
        self.where_result(shape_format)

    def test_where_shape_format_fp32_2d(self, device="npu"):
        format_list = [0]
        shape_format = [[np.float32, i, [5, 256]] for i in format_list]
        self.where_result(shape_format)

    def test_where_shape_format_fp32_3d(self, device="npu"):
        format_list = [0]
        shape_format = [[np.float32, i, [32, 3, 3]] for i in format_list]
        self.where_result(shape_format)

    def test_where_shape_format_fp32_4d(self, device="npu"):
        format_list = [0, 3]
        shape_format = [[np.float32, i, [64, 112, 7, 7]] for i in format_list]
        self.where_result(shape_format)

    def test_where_shape_format_fp16_1d(self, device="npu"):
        format_list = [0, 3]
        shape_format = [[np.float16, i, [18]] for i in format_list]
        self.where_result(shape_format)

    def test_where_shape_format_fp16_2d(self, device="npu"):
        format_list = [0, 3, 4, 29]
        shape_format = [[np.float16, i, [5, 256]] for i in format_list]
        self.where_result(shape_format)

    def test_where_shape_format_fp16_3d(self, device="npu"):
        format_list = [0, 3, 4, 29]
        shape_format = [[np.float16, i, [32, 3, 3]] for i in format_list]
        self.where_result(shape_format)

    def test_where_shape_format_fp16_4d(self, device="npu"):
        format_list = [0, 3, 4, 29]
        shape_format = [[np.float16, i, [64, 112, 7, 7]] for i in format_list]
        self.where_result(shape_format)


if __name__ == "__main__":
    run_tests()
