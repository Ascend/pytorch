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

class TestAll(TestCase):
    def cpu_op_exec1(self, input):
        output = torch.all(input)
        output = output.numpy()
        return output
    
    def npu_op_exec1(self, input):
        input = input.to("npu")
        output = torch.all(input)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec3(self, input, axis, keepdim):
        output = torch.all(input, axis, keepdim)
        output = output.numpy()
        return output
    
    def npu_op_exec3(self, input, axis, keepdim):
        input = input.to("npu")
        output = torch.all(input, axis, keepdim)
        output = output.to("cpu")
        output = output.numpy()
        return output
    
    def test_all_noaxis_notkeedim_bool(self, device):
        shape_format = [
            [np.bool_, -1, (4, 2, 5)],
            [np.bool_, -1, (7, 4, 5, 8)]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 2)
            cpu_output = self.cpu_op_exec1(cpu_input)
            npu_output = self.npu_op_exec1(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_all_axis_notkeedim_bool(self, device):
        shape_format = [
            [[np.bool_, -1, (4, 2, 5)], 3],
            [[np.bool_, -1, (4, 2, 5, 8)], 4]
        ]
        for item in shape_format:
            for i in range(item[1]):
                cpu_input, npu_input = create_common_tensor(item[0], 0, 2)
                cpu_output = self.cpu_op_exec3(cpu_input, i, False)
                npu_output = self.npu_op_exec3(npu_input, i, False)
                self.assertRtolEqual(cpu_output, npu_output)

    def test_all_axis_keedim_bool(self, device):
        shape_format = [
            [[np.bool_, -1, (4, 2, 5)], 3],
            [[np.bool_, -1, (4, 2, 5, 8)], 4]
        ]
        for item in shape_format:
            for i in range(item[1]):
                cpu_input, npu_input = create_common_tensor(item[0], 0, 2)
                cpu_output = self.cpu_op_exec3(cpu_input, i, True)
                npu_output = self.npu_op_exec3(npu_input, i, True)
                self.assertRtolEqual(cpu_output, npu_output)   

instantiate_device_type_tests(TestAll, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()