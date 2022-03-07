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
import torch_npu
import numpy as np

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor

class TestBernoulli(TestCase):
    def cpu_op_exec(self, input1):
        output = torch.bernoulli(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.bernoulli(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_inplace_tensor_exec(self, input1, p):
        output = input1.bernoulli_(p)
        output = output.numpy()
        return output

    def npu_op_inplace_tensor_exec(self, input1, p):
        output = input1.bernoulli_(p)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_inplace_float_exec(self, input1):
        output = input1.bernoulli_(0.5)
        output = output.numpy()
        return output

    def npu_op_inplace_float_exec(self, input1):
        output = input1.bernoulli_(0.5)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_out_exec(self, input1, output1):
        torch.bernoulli(input1, out=output1)
        output1 = output1.numpy()
        return output1

    def npu_op_out_exec(self, input1, output1):
        torch.bernoulli(input1, out=output1)
        output1 = output1.to("cpu")
        output1 = output1.numpy()
        return output1

    def cpu_op_p_exec(self, input1, p):
        input1 = input1.bernoulli_(p)
        input1 = input1.numpy()
        return input1

    def npu_op_p_exec(self, input1, p):
        input1 = input1.bernoulli_(p)
        input1 = input1.to("cpu")
        input1 = input1.numpy()
        return input1

    def test_bernoulli_p(self):
        format_list = [0, 3]
        shape_list = [(2, 3, 4)]
        p_list = [0.2, 0.6]
        shape_format = [
            [np.float32, i, j, k] for i in format_list for j in shape_list
            for k in p_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 1)
            cpu_output = self.cpu_op_p_exec(cpu_input, item[3])
            npu_output = self.npu_op_p_exec(npu_input, item[3])
            print(cpu_output, npu_output)


    def test_bernoulli_out_float32(self):
        format_list = [0, 3]
        shape_list = [(2, 3, 4)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 1)
            cpu_output1, npu_output1 = create_common_tensor(item, 0, 1)
            cpu_output = self.cpu_op_out_exec(cpu_input, cpu_output1)
            npu_output = self.npu_op_out_exec(npu_input, npu_output1)
            print(cpu_output, npu_output)

    def test_bernoulli_out_float16(self):
        format_list = [0, 3]
        shape_list = [(2, 3, 4)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 1)
            cpu_output1, npu_output1 = create_common_tensor(item, 0, 1)
            cpu_input = cpu_input.to(torch.float32)
            cpu_output1 = cpu_output1.to(torch.float32)
            cpu_output = self.cpu_op_out_exec(cpu_input, cpu_output1)
            npu_output = self.npu_op_out_exec(npu_input, npu_output1)
            cpu_output = cpu_output.astype(np.float16)
            print(cpu_output, npu_output)

    def test_bernoulli_float32(self):
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

    def test_bernoulli_float16(self):
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

    def test_bernoulli_tensor_p(self):
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

    def test_bernoulli_float_p(self):
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

if __name__ == '__main__':
    run_tests()