# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION. 
# All rights reserved.
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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestTril(TestCase):
    def test_tril(self, device):
        dtype_list = [np.float32, np.float16]
        format_list = [0, 3, 4]
        shape_list = [[5, 5],[4, 5, 6]]
        diagonal_list = [-1, 0, 1]
        shape_format = [
            [i, j, k, l] for i in dtype_list for j in format_list for k in shape_list for l in diagonal_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[:-1], 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input, item[-1])
            npu_output = self.npu_op_exec(npu_input, item[-1])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_tril_inplace(self, device):
        dtype_list = [np.float32, np.float16]
        format_list = [0, 3, 4]
        shape_list = [[5, 5], [4, 5, 6]]
        diagonal_list = [-1, 0, 1]
        shape_format = [
            [i, j, k, l] for i in dtype_list for j in format_list for k in shape_list for l in diagonal_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[:-1], 0, 100)
            cpu_output = self.cpu_op_inplace_exec(cpu_input, item[-1])
            npu_output = self.npu_op_inplace_exec(npu_input, item[-1])
            self.assertRtolEqual(cpu_output, npu_output)

    def cpu_op_exec(self, input, diagonal=0):
        output = torch.tril(input, diagonal)
        output = output.numpy()
        return output

    def npu_op_exec(self, input, diagonal=0):
        output = torch.tril(input, diagonal)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_inplace_exec(self, input, diagonal=0):
        output = input.tril_(diagonal)
        output = output.numpy()
        return output

    def npu_op_inplace_exec(self, input, diagonal=0):
        output = input.tril_(diagonal)
        output = output.to("cpu")
        output = output.numpy()
        return output

instantiate_device_type_tests(TestTril, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
