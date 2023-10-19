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


class TestLog10(TestCase):
    def cpu_op_exec(self, input1):
        output = torch.log10(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.log10(input1)
        output = output.to("cpu").numpy()
        return output

    def npu_op_exec_out(self, input1, input2):
        torch.log10(input1, out=input2)
        output = input2.to("cpu").numpy()
        return output

    def cpu_inp_op_exec(self, input1):
        output = torch.log10_(input1)
        output = output.numpy()
        return output

    def npu_inp_op_exec(self, input1):
        torch.log10_(input1)
        output = input1.to("cpu").numpy()
        return output

    def cpu_inp_uncon_op_exec(self, input1):
        input1 = input1.as_strided([2, 2], [1, 2], 2)
        output = torch.log10_(input1)
        output = output.numpy()
        return output

    def npu_inp_uncon_op_exec(self, input1):
        input1 = input1.as_strided([2, 2], [1, 2], 2)
        torch.log10_(input1)
        output = input1.to("cpu").numpy()
        return output

    def test_log10_shape_format_fp32(self, device="npu"):
        format_list = [3]
        shape_list = [(4, 4)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            cpu_output1 = self.cpu_inp_op_exec(cpu_input1)
            npu_output1 = self.npu_inp_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output1, npu_output1)

    def test_log10_shape_format_fp16(self, device="npu"):
        format_list = [3]
        shape_list = [(4, 4)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            cpu_output = cpu_output.astype(np.float16)
            cpu_output1 = self.cpu_inp_op_exec(cpu_input1)
            npu_output1 = self.npu_inp_op_exec(npu_input1)
            cpu_output1 = cpu_output1.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output1, npu_output1)

    def test_log10_inp_uncon_shape_format_fp32(self, device="npu"):
        format_list = [3]
        shape_list = [(8, 6)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_output = self.cpu_inp_uncon_op_exec(cpu_input1)
            npu_output = self.npu_inp_uncon_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_log10_inp_uncon_shape_format_fp16(self, device="npu"):
        format_list = [3]
        shape_list = [(8, 6)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_inp_uncon_op_exec(cpu_input1)
            npu_output = self.npu_inp_uncon_op_exec(npu_input1)
            cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_log10_out_float32_shape_format(self, device="npu"):
        shape_format = [
            [[np.float32, 0, [1024, 32, 7, 7]], [np.float32, 0, [1024, 32, 7, 7]]],
            [[np.float32, 0, [1024, 32, 7]], [np.float32, 0, [1024, 32]]],
            [[np.float32, 0, [1024, 32]], [np.float32, 0, [1024, 32]]],
            [[np.float32, 0, [1024]], [np.float32, 0, [1024, 1]]],
            [[np.float32, 3, [1024, 32, 7, 7]], [np.float32, 3, [1024, 32, 7, 7]]],
            [[np.float32, 3, [1024, 32, 7]], [np.float32, 3, [1024, 32]]],
            [[np.float32, 3, [1024, 32]], [np.float32, 3, [1024, 20]]],
            [[np.float32, 3, [1024]], [np.float32, 3, [1024]]],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            cpu_output, npu_output = create_common_tensor(item[1], 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec_out(npu_input, npu_output)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_log10_out_float16_shape_format(self, device="npu"):
        shape_format = [
            [[np.float16, 0, [1024, 32, 7, 7]], [np.float16, 0, [1024, 32, 7, 7]]],
            [[np.float16, 0, [1024, 32, 7]], [np.float16, 0, [1024, 32]]],
            [[np.float16, 0, [1024, 32]], [np.float16, 0, [1024, 32]]],
            [[np.float16, 0, [1024]], [np.float16, 0, [1024, 1]]],
            [[np.float16, 3, [1024, 32, 7, 7]], [np.float16, 3, [1024, 32, 7, 7]]],
            [[np.float16, 3, [1024, 32, 7]], [np.float16, 3, [1024, 32]]],
            [[np.float16, 3, [1024, 32]], [np.float16, 3, [1024, 20]]],
            [[np.float16, 3, [1024]], [np.float16, 3, [1024]]],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            cpu_output, npu_output = create_common_tensor(item[1], 0, 100)
            if item[0][0] == np.float16:
                cpu_input = cpu_input.to(torch.float32)
                cpu_output = cpu_output.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec_out(npu_input, npu_output)
            if item[0][0] == np.float16:
                cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
