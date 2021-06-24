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
from common_device_type import instantiate_device_type_tests
from util_test import create_common_tensor


class TestFillDiagonal(TestCase):
    def npu_op_exec(self, input):
        input = input.npu()
        input.fill_diagonal_(1)
        output = input.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec(self, input):
        input.fill_diagonal_(1)
        output = input.numpy()
        return output

    def npu_op_wrap_exec(self, input):
        input = input.npu()
        input.fill_diagonal_(1, wrap=True)
        output = input.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_wrap_exec(self, input):
        input.fill_diagonal_(1, wrap=True)
        output = input.numpy()
        return output

    def test_fill_diagonal_shape_format_fp32(self, device):
        format_list = [0, 3]
        shape_list = ([7, 3], [3, 3, 3])
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_input1 = cpu_input.clone()
            npu_input1 = npu_input.clone()
            cpu_output1 = self.cpu_op_exec(cpu_input)
            npu_output1 = self.npu_op_exec(npu_input)
            cpu_output2 = self.cpu_op_wrap_exec(cpu_input1)
            npu_output2 = self.npu_op_wrap_exec(npu_input1)
            self.assertRtolEqual(cpu_output1, npu_output1)
            self.assertRtolEqual(cpu_output2, npu_output2)

    def test_fill_diagonal_shape_format_fp16(self, device):
        format_list = [0, 3]
        shape_list = ([7, 3], [3, 3, 3])
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_input1 = cpu_input.clone()
            npu_input1 = npu_input.clone()
            cpu_output1 = self.cpu_op_exec(cpu_input)
            npu_output1 = self.npu_op_exec(npu_input)
            cpu_output2 = self.cpu_op_wrap_exec(cpu_input1)
            npu_output2 = self.npu_op_wrap_exec(npu_input1)
            self.assertRtolEqual(cpu_output1, npu_output1)
            self.assertRtolEqual(cpu_output2, npu_output2)


instantiate_device_type_tests(TestFillDiagonal, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
