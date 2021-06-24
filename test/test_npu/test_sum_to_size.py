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

import torch
import numpy as np
import sys
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestSumToSize(TestCase):

    def generate_single_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input1 = torch.from_numpy(input1)
        return input1

    def cpu_op_exec(self, input1, shape):
        output = input1.sum_to_size(shape)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, shape):
        input1 = input1.to("npu")
        output = input1.sum_to_size(shape)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_sum_to_size_float16(self, device):
        def cpu_op_exec_fp16(input1, shape):
            input1 = input1.to(torch.float32)
            output = input1.sum_to_size(shape)
            output = output.numpy()
            output = output.astype(np.float16)
            return output
        input1 = self.generate_single_data(0, 100, (5,3), np.float16)
        cpu_output = cpu_op_exec_fp16(input1, (5,1))
        npu_output = self.npu_op_exec(input1, (5,1))
        self.assertRtolEqual(cpu_output, npu_output)    

    def test_sum_to_size_float32_two(self, device):
        input1 = self.generate_single_data(0, 100, (4,3), np.float32)
        cpu_output = self.cpu_op_exec(input1, (4,1))
        npu_output = self.npu_op_exec(input1, (4,1))
        self.assertRtolEqual(cpu_output, npu_output)

    def test_sum_to_size_float32_three(self, device):
        input1 = self.generate_single_data(0, 100, (4,3,6), np.float32)
        cpu_output = self.cpu_op_exec(input1, (4,3,1))
        npu_output = self.npu_op_exec(input1, (4,3,1))
        self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestSumToSize, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:3")
    run_tests()
