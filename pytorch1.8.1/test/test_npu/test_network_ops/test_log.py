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
import sys
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestLog(TestCase):
    def cpu_op_exec(self, input1):
        output = torch.log(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        input1 = input1.to("npu")
        output = torch.log(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1):
        input1 = input1.to("npu")
        output = input1.to("npu")
        torch.log(input1, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_inp_op_exec(self, input1):
        output = torch.log_(input1)
        output = output.numpy()
        return output

    def npu_inp_op_exec(self, input1):
        input1 = input1.to("npu")
        output = torch.log_(input1)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def cpu_inp_uncon_op_exec(self, input1):
        input1 = input1.as_strided([2, 2], [1, 2], 2)
        output = torch.log_(input1)
        output = output.numpy()
        return output

    def npu_inp_uncon_op_exec(self, input1):
        input1 = input1.to("npu")
        input1 = input1.as_strided([2, 2], [1, 2], 2)
        output = torch.log_(input1)
        output = input1.to("cpu")
        output = output.numpy()
        return output

# TestCase
    def test_log_shape_format_fp32(self, device):
        format_list = [3]
        shape_list = [(4, 4)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_log_shape_format_fp16(self, device):
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
            self.assertRtolEqual(cpu_output, npu_output)

    def test_log_inp_shape_format_fp32(self, device):
        format_list = [3]
        shape_list = [(4, 4)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_output = self.cpu_inp_op_exec(cpu_input1)
            npu_output = self.npu_inp_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_log_inp_shape_format_fp16(self, device):
        format_list = [3]
        shape_list = [(4, 4)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_inp_op_exec(cpu_input1)
            npu_output = self.npu_inp_op_exec(npu_input1)
            cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_log_inp_uncon_shape_format_fp32(self, device):
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

    def test_log_inp_uncon_shape_format_fp16(self, device):
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
            print("cpu:", cpu_output, "npu:", npu_output)
            self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestLog, globals(), except_for="cpu")
if __name__ == '__main__':
    run_tests()
