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

class TestEqual(TestCase):

    def cpu_op_exec(self, input1, input2):
        output =torch.equal(input1, input2)
        output = np.array([output], dtype = bool)
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.equal(input1, input2)
        output = np.array([output], dtype = bool)
        return output

    def test_equal_common_shape_format(self, device):
        shape_format = [
                [[np.float32, -1, (5, 3)], [np.float32, -1, (5, 3)]],
                [[np.int32, -1, (4, 3)], [np.int32, -1, (4, 3)]],
                [[np.int8, -1, (8, 8)], [np.int8, -1, (8, 8)]],
                [[np.uint8, -1, (8, 8)], [np.uint8, -1, (8, 8)]],
                [[np.float32, -1, (5, 3, 12)], [np.float32, -1, (5, 3, 12)]],
                [[np.float32, -1, (4, 3, 100)], [np.float32, -1, (4, 3, 100)]],
                [[np.float32, -1, (8, 8, 12, 12, 10)], [np.float32, -1, (8, 8, 12, 12, 10)]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)
    
    def test_equal_common_shape_fp16_format(self, device):
        def cpu_op_fp16_exec(input1, input2):
            input1 = input1.to(torch.float32)
            input2 = input2.to(torch.float32)
            output = torch.equal(input1, input2)
            output = np.array([output], dtype = bool)
            return output
        shape_format = [
                [[np.float16, -1, (5, 3, 12)], [np.float16, -1, (5, 3, 12)]],
                [[np.float16, -1, (4, 3, 100)], [np.float16, -1, (4, 3, 100)]],
                [[np.float16, -1, (8, 8, 12, 12, 10)], [np.float16, -1, (8, 8, 12, 12, 10)]]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            cpu_output = cpu_op_fp16_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestEqual, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
