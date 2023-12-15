#  Copyright (c) 2020, Huawei Technologies.All rights reserved.
#  Licensed under the BSD 3-Clause License  (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  https://opensource.org/licenses/BSD-3-Clause
#
#  Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import itertools
import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestTrueDivide(TestCase):

    def cpu_op_exec(self, input1, input2):
        output = torch.true_divide(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.true_divide(input1, input2)
        output = output.cpu().numpy()
        return output

    def cpu_op_out_exec(self, input1, input2, output):
        output = torch.true_divide(input1, input2, out=output)
        output = output.numpy()
        return output

    def npu_op_out_exec(self, input1, input2, output):
        output = torch.true_divide(input1, input2, out=output)
        output = output.cpu().numpy()
        return output

    def cpu_op_exec_inplace(self, input1, input2):
        input1.true_divide_(input2)
        input1 = input1.numpy()
        return input1

    def npu_op_exec_inplace(self, input1, input2):
        input1.true_divide_(input2)
        input1 = input1.cpu().numpy()
        return input1

    def compare_without_nan(self, cpu_output, npu_output):
        mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
        self.assertRtolEqual(cpu_output[mask], npu_output[mask])

    def true_divide_result(self, cpu_input1, cpu_input2, npu_input1, npu_input2):
        cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
        npu_output = self.npu_op_exec(npu_input1, npu_input2)
        self.compare_without_nan(cpu_output, npu_output)

    def true_divide_out_result(self, cpu_input1, cpu_input2, npu_input1, npu_input2):
        output = torch.randn(5, 3).uniform_(-100, 100).to(torch.float32)
        cpu_output = self.cpu_op_out_exec(cpu_input1, cpu_input2, output)
        npu_output = self.npu_op_out_exec(npu_input1, npu_input2, output.npu())
        self.compare_without_nan(cpu_output, npu_output)

    def true_divide_inplace_result(self, cpu_input1, cpu_input2, npu_input1, npu_input2):
        cpu_output = self.cpu_op_exec_inplace(cpu_input1, cpu_input2)
        npu_output = self.npu_op_exec_inplace(npu_input1, npu_input2)
        self.compare_without_nan(cpu_output, npu_output)

    def test_true_divide_tensor(self):
        shape_format = [
            [[np.int32, 0, (2, 2)], [np.int32, 0, (2)]],
            [[np.float32, 0, (2, 2)], [np.float32, 0, (2)]],
            [[np.float16, 0, (2, 2)], [np.float16, 0, (2)]],
            [[np.int32, 0, (4, 3)], [np.int32, 0, (4, 3)]],
            [[np.float32, 0, (4, 3)], [np.float32, 0, (4, 3)]],
            [[np.float16, 0, (4, 3)], [np.float16, 0, (4, 3)]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            self.true_divide_result(cpu_input1, cpu_input2, npu_input1, npu_input2)

    def test_true_divide_bool(self):
        dtype_list = [np.int32, np.float16, np.float32]
        for item in dtype_list:
            cpu_input1, npu_input1 = create_common_tensor([item, 0, (2, 2)], 0, 10)
            cpu_input2, npu_input2 = create_common_tensor([item, 0, (2, 2)], 5, 10)
            self.true_divide_result(cpu_input1 > 5, 1.2, npu_input1 > 5, 1.2)
            self.true_divide_result(cpu_input1, cpu_input2 > 5, npu_input1, npu_input2 > 5)
            self.true_divide_result(cpu_input1 > 5, cpu_input2 > 5, npu_input1 > 5, npu_input2 > 5)

    def test_true_divide_scalar(self):
        input_list = [True, 2, 2.0]
        shape_list = [(2, 3), (2, 2)]
        dtype_list = [np.float16, np.float32]
        for item in itertools.product(dtype_list, shape_list, input_list):
            cpu_input1, npu_input1 = create_common_tensor([item[0], 0, item[1]], 0, 100)
            self.true_divide_result(cpu_input1, item[2], npu_input1, item[2])

    def test_true_divide_out_tensor(self):
        shape_format = [
            [[np.int32, 0, (2, 2)], [np.int32, 0, (2)]],
            [[np.float32, 0, (2, 2)], [np.float32, 0, (2)]],
            [[np.float16, 0, (2, 2)], [np.float16, 0, (2)]],
            [[np.int32, 0, (5, 3, 2, 4)], [np.int32, 0, (5, 3, 2, 4)]],
            [[np.float32, 0, (5, 3, 2, 4)], [np.float32, 0, (5, 3, 2, 4)]],
            [[np.float16, 0, (5, 3, 2, 4)], [np.float16, 0, (5, 3, 2, 4)]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            self.true_divide_out_result(cpu_input1, cpu_input2, npu_input1, npu_input2)

    def test_true_divide_out_bool(self):
        dtype_list = [np.int32, np.float16, np.float32]
        for item in dtype_list:
            cpu_input1, npu_input1 = create_common_tensor([item, 0, (2, 2)], 0, 10)
            cpu_input2, npu_input2 = create_common_tensor([item, 0, (2, 2)], 5, 10)
            self.true_divide_out_result(cpu_input1 > 5, 1.2, npu_input1 > 5, 1.2)
            self.true_divide_out_result(cpu_input1, cpu_input2 > 5, npu_input1, npu_input2 > 5)
            self.true_divide_out_result(cpu_input1 > 5, cpu_input2 > 5, npu_input1 > 5, npu_input2 > 5)

    def test_true_divide_out_scalar(self):
        input_list = [True, 2, 2.0]
        dtype_list = [np.int32, np.float16, np.float32]
        for item in itertools.product(dtype_list, input_list):
            cpu_input1, npu_input1 = create_common_tensor([item[0], 0, (2, 2)], 0, 100)
            self.true_divide_out_result(cpu_input1, item[1], npu_input1, item[1])

    def test_true_divide_inplace_tensor(self):
        shape_format = [
            [[np.float32, 0, (2, 2)], [np.float32, 0, (2)]],
            [[np.float16, 0, (2, 2)], [np.float16, 0, (2)]],
            [[np.float32, 0, (5, 3, 2, 4)], [np.float32, 0, (5, 3, 2, 4)]],
            [[np.float16, 0, (5, 3, 2, 4)], [np.float16, 0, (5, 3, 2, 4)]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            self.true_divide_inplace_result(cpu_input1, cpu_input2, npu_input1, npu_input2)

    def test_true_divide_inplace_bool(self):
        dtype_list = [np.float16, np.float32]
        for item in dtype_list:
            cpu_input1, npu_input1 = create_common_tensor([item, 0, (5, 3, 2, 4)], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor([item, 0, (5, 3, 2, 4)], 0, 100)
            self.true_divide_inplace_result(cpu_input1, cpu_input2 > 5, npu_input1, npu_input2 > 5)

    def test_true_divide_inplace_scalar(self):
        input_list = [True, 2, 2.0]
        dtype_list = [np.float16, np.float32]
        for item in itertools.product(dtype_list, input_list):
            cpu_input1, npu_input1 = create_common_tensor([item[0], 0, (5, 3, 2, 4)], 0, 100)
            self.true_divide_inplace_result(cpu_input1, item[1], npu_input1, item[1])

    def test_true_divide_scalar_input(self):
        format_shape = [
            [[np.float32, 0, 1], 2],
            [[np.float16, 0, (1, 4)], 4.0],
            [[np.bool_, 0, (1, 29126, 10)], 10],
            [[np.int64, 0, (13, 132)], True],
            [[np.int32, 0, (32, 12)], 5],
        ]
        for item in format_shape:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            self.true_divide_result(item[1], cpu_input1, item[1], npu_input1)
            self.true_divide_out_result(item[1], cpu_input1, item[1], npu_input1)


if __name__ == "__main__":
    run_tests()
