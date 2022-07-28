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

class TestBernoulli(TestCase):
    def get_common_tensor(self, item, minValue, maxValue):
        dtype = item[0]
        shape = item[2]
        input1 = np.random.uniform(minValue, maxValue, shape).astype(dtype)
        npu_input = torch.from_numpy(input1).to("npu")
        return npu_input

    def npu_op_exec(self, input1):
        torch.manual_seed(4)
        output = torch.bernoulli(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_inplace_tensor_exec(self, input1, p):
        torch.manual_seed(4)
        output = input1.bernoulli_(p)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_inplace_float_exec(self, input1):
        torch.manual_seed(4)
        output = input1.bernoulli_(0.5)
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
            npu_input = self.get_common_tensor(item, 0, 1)
            npu_expect_output = self.npu_op_exec(npu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertEqual(npu_expect_output, npu_output)
            
    def test_bernoulli_float16(self, device):
        format_list = [0, 3]
        shape_list = [(2, 3, 4)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            npu_input = self.get_common_tensor(item, 0, 1)
            npu_expect_output = self.npu_op_exec(npu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertEqual(npu_expect_output, npu_output)

    def test_bernoulli_tensor_p(self, device):
        format_list = [0, 3]
        shape_list = [(2, 3, 4)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            npu_input = self.get_common_tensor(item, 0, 1)
            npu_input_p = self.get_common_tensor(item, 0, 1)
            npu_expect_output = self.npu_op_inplace_tensor_exec(npu_input, npu_input_p)
            npu_output = self.npu_op_inplace_tensor_exec(npu_input, npu_input_p)
            self.assertEqual(npu_expect_output, npu_output)

    def test_bernoulli_float_p(self, device):
        format_list = [0, 3]
        shape_list = [(2, 3, 4)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            npu_input = self.get_common_tensor(item, 0, 1)
            npu_expect_output = self.npu_op_inplace_float_exec(npu_input)
            npu_output = self.npu_op_inplace_float_exec(npu_input)
            self.assertEqual(npu_expect_output, npu_output)

instantiate_device_type_tests(TestBernoulli, globals(), except_for="cpu")
if __name__ == '__main__':
    run_tests()

