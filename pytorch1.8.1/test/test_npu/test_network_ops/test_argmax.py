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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestArgmax(TestCase):
    def cpu_op_exec(self, input):
        output = torch.argmax(input)
        output = output.numpy()
        return output

    def npu_op_exec(self, input):
        output = torch.argmax(input)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_argmax_shape_format_fp16(self, device):
        format_list = [0]
        shape_list = [[5], [2, 4], [2, 2, 4], [2, 3, 3, 4]]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -10, 10)
            cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_argmax_shape_format_fp32(self, device):
        format_list = [0]
        shape_list = [[5], [2, 4], [2, 2, 4], [2, 3, 3, 4]]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -10, 10)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def cpu_op_exec1(self, input, dim):
        output = torch.argmax(input, dim)
        output = output.numpy()
        return output

    def npu_op_exec1(self, input, dim):
        output = torch.argmax(input, dim)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_argmaxd_shape_format_fp16(self, device):
        format_list = [0]
        shape_list = [[5], [2, 4], [2, 2, 4], [2, 3, 3, 4]]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -10, 10)
            cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_exec1(cpu_input, -1)
            npu_output = self.npu_op_exec1(npu_input, -1)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_argmaxd_shape_format_fp32(self, device):
        format_list = [0]
        shape_list = [[5], [2, 4], [2, 2, 4], [2, 3, 3, 4]]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -10, 10)
            cpu_output = self.cpu_op_exec1(cpu_input, -1)
            npu_output = self.npu_op_exec1(npu_input, -1)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestArgmax, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
