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


class TestNotEqual(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = torch.not_equal(input1, input2)
        output = output.numpy()
        return output

    def npu_op_inplace_exec(self, input1, input2):
        input1.not_equal_(input2)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_inplace_exec(self, input1, input2):
        input1.not_equal_(input2)
        output = input1.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.not_equal(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2):
        input3 = torch.empty(0).bool().npu()
        torch.not_equal(input1, input2, out=input3)
        output = input3.to("cpu")
        output = output.numpy()
        return output

    def test_not_equal_shape_format_fp32(self):
        dtype_list = [np.float32]
        format_list = [0, 3]
        shape_list = [[1024], [8, 128], [2, 8, 128], [2, 8, 128, 512]]
        shape_format = [
            [d, i, j] for d in dtype_list for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertEqual(cpu_output, npu_output)

    def test_not_equal_shape_format_fp16(self):
        dtype_list = [np.float16]
        format_list = [0, 3]
        shape_list = [[1024], [8, 128], [2, 8, 128], [2, 8, 128, 512]]
        shape_format = [
            [d, i, j] for d in dtype_list for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 1, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertEqual(cpu_output, npu_output)

    def test_not_equal_inp_shape_format(self):
        dtype_list = [np.float16, np.float32]
        format_list = [0, 3]
        shape_list = [[1024], [8, 128], [2, 8, 128], [2, 8, 128, 512]]
        shape_format = [
            [d, i, j] for d in dtype_list for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 1, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_inplace_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_inplace_exec(npu_input1, npu_input2)
            self.assertEqual(cpu_output, npu_output)

    def test_not_equal_inp_scalar_shape_format(self):
        dtype_list = [np.float16, np.float32]
        format_list = [0, 3]
        shape_list = [[1024], [8, 128], [2, 8, 128], [2, 8, 128, 512]]
        shape_format = [
            [d, i, j] for d in dtype_list for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 1, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_inplace_exec(cpu_input1, 5)
            npu_output = self.npu_op_inplace_exec(npu_input1, 5)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertEqual(cpu_output, npu_output)

    def test_not_equal_out_shape_format_fp32(self):
        dtype_list = [np.float32]
        format_list = [0]
        shape_list = [[1024], [8, 128], [2, 8, 128], [2, 8, 128, 512]]
        shape_format = [
            [[d, i, j]] for d in dtype_list for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -10, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[0], -10, 10)
            npu_output_out = self.npu_op_exec_out(npu_input1, npu_input2)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            self.assertEqual(cpu_output, npu_output_out)

    def test_not_equal_scalar_out_shape_format_fp32(self):
        dtype_list = [np.float32]
        format_list = [0]
        shape_list = [[1024], [8, 128], [2, 8, 128], [2, 8, 128, 512]]
        shape_format = [
            [[d, i, j]] for d in dtype_list for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -10, 10)
            npu_output_out = self.npu_op_exec_out(npu_input1, 5)
            cpu_output = self.cpu_op_exec(cpu_input1, 5)
            self.assertEqual(cpu_output, npu_output_out)

    def test_not_equal_mix_dtype(self):
        cpu_input1, npu_input1 = create_common_tensor([np.float16, 0, (2, 3)], 1, 100)
        cpu_input2, npu_input2 = create_common_tensor([np.float32, 0, (2, 3)], 1, 100)
        cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
