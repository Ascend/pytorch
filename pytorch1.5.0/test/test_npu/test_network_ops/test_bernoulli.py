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
import copy
import sys
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestBernoulli(TestCase):
    def cpu_op_exec(self, input):
        output = torch.bernoulli(input)
        output = output.numpy()
        return output

    def npu_op_exec(self, input):
        output = torch.bernoulli(input)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_inplace_tensor_exec(self, input, p):
        output = input.bernoulli_(p)
        output = output.numpy()
        return output

    def npu_op_inplace_tensor_exec(self, input, p):
        output = input.bernoulli_(p)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_inplace_float_exec(self, input):
        output = input.bernoulli_(0.5)
        output = output.numpy()
        return output

    def npu_op_inplace_float_exec(self, input):
        output = input.bernoulli_(0.5)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_bernoulli_float32(self, device):
        format_list = [0, 3]
        shape_list = [(2, 3, 4)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 1)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            print(cpu_output, npu_output)
            #self.assertEqual(cpu_output, npu_output)
            #生成随机值，无法对比cpu值

    def test_bernoulli_float16(self, device):
        format_list = [0, 3]
        shape_list = [(2, 3, 4)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 1)
            cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            cpu_output = cpu_output.astype(np.float16)
            print(cpu_output, npu_output)
            #self.assertEqual(cpu_output, npu_output)

    def test_bernoulli_tensor_p(self, device):
        format_list = [0, 3]
        shape_list = [(2, 3, 4)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 1)
            cpu_input_p, npu_input_p = create_common_tensor(item, 0, 1)
            cpu_output = self.cpu_op_inplace_tensor_exec(cpu_input, cpu_input_p)
            npu_output = self.npu_op_inplace_tensor_exec(npu_input, npu_input_p)
            print(cpu_output, npu_output)
            #self.assertEqual(cpu_output, npu_output)

    def test_bernoulli_float_p(self, device):
        format_list = [0, 3]
        shape_list = [(2, 3, 4)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 1)
            cpu_output = self.cpu_op_inplace_float_exec(cpu_input)
            npu_output = self.npu_op_inplace_float_exec(npu_input)
            print(cpu_output, npu_output)
            #self.assertEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestBernoulli, globals(), except_for="cpu")
if __name__ == '__main__':
    run_tests()

