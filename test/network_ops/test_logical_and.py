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


class TestLogicalAnd(TestCase):
    def generate_single_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        return npu_input1

    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
        return npu_input1, npu_input2

    def generate_three_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input3 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
        npu_input3 = torch.from_numpy(input3)
        return npu_input1, npu_input2, npu_input3

    def cpu_op_exec(self, input1, input2):
        output = torch.logical_and(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        output = torch.logical_and(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_out(self, input1, input2, input3):
        torch.logical_and(input1, input2, out=input3)
        output = input3.numpy()
        return output

    def npu_op_exec_out(self, input1, input2, input3):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        output = input3.to("npu")
        torch.logical_and(input1, input2, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_(self, input1, input2):
        output = torch.Tensor.logical_and_(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec_(self, input1, input2):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        output = torch.Tensor.logical_and_(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def logical_and_out_result(self, shape_format):
        for item in shape_format:
            cpu_input1 = torch.randn(item[0]) < 0
            cpu_input2 = torch.randn(item[0]) < 0
            cpu_input3 = torch.randn(item[1]) < 0
            cpu_output_out = self.cpu_op_exec_out(cpu_input1, cpu_input2, cpu_input3)
            npu_output_out = self.npu_op_exec_out(cpu_input1, cpu_input2, cpu_input3)
            self.assertRtolEqual(cpu_output_out, npu_output_out)

    def test_logical_and_out(self, device="npu"):
        shape_format = [
            [[128, 116, 14, 14], [256, 116, 1, 1, 28]],
            [[128, 3, 224, 224], [3, 3, 3]],
            [[128, 116, 14, 14], [128, 116, 14, 14]],
            [[256, 128, 7, 7], [128, 256, 3, 3, 28]],
            [[2, 3, 3, 3], [3, 1, 3]],
            [[128, 232, 7, 7], [128, 232, 7, 7]],
        ]
        self.logical_and_out_result(shape_format)

    def test_logical_and_bool(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(0, 2, (2, 5), np.bool)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input2).astype(np.float32)
        npu_output = self.npu_op_exec(npu_input1, npu_input2).astype(np.float32)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_logical_and_inplace_bool(self, device="npu"):
        npu_input1, npu_input2 = self.generate_data(0, 2, (2, 5), np.bool)
        cpu_output = self.cpu_op_exec_(npu_input1, npu_input2).astype(np.float32)
        npu_output = self.npu_op_exec_(npu_input1, npu_input2).astype(np.float32)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
