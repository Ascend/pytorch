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

class TestDropout(TestCase):

    def cpu_op_exec(self, input1, p, training, inplace):
        output = torch.nn.functional.dropout(input1, p, training, inplace)
        output = output.numpy()
        return output
    
    def npu_op_extc(self, input1, p, training, inplace):
        output = torch.nn.functional.dropout(input1, p, training, inplace)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_dropout_common_shape_format(self, device):
        shape_format = [
            [[np.float32, -1, (2, 2)], 0.5, True, False],
            [[np.float32, -1, (2, 2, 2)], 0.5, True, False],
            [[np.float32, -1, (2, 2, 2,5)], 0.5,False, True],
            [[np.float32, -1, (10)], 0.5, False, False],
            [[np.float32, -1, (1, 2)], 0.6, False, True],
            [[np.float32, -1, (12, 24, 1024)], 0.7, False, False],
            [[np.float32, -1, (16, 8, 12, 15)], 0.2, False, True]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 1)
            cpu_input2, npu_input2 = item[1], item[1]
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, item[2], item[3])
            npu_output = self.npu_op_extc(npu_input1, npu_input2, item[2], item[3])
            #self.assertRtolEqual(cpu_output, npu_output)
    
    def test_dropout_float16_shape_format(self, device):
        def cpu_op_exec_fp16(input1, p, training, inplace):
            input1 = input1.to(torch.float32)
            output = torch.nn.functional.dropout(input1, p, training, inplace)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        shape_format = [
            [[np.float16, -1, (2, 2)], 0.5, True, False],
            [[np.float16, -1, (2, 2, 2)], 0.5, True, False],
            [[np.float16, -1, (2, 2, 2,5)], 0.5,False, True],
            [[np.float16, -1, (10)], 0.5, False, False],
            [[np.float16, -1, (1, 2)], 0.6, False, True],
            [[np.float16, -1, (12, 24, 1024)], 0.7, False, False],
            [[np.float16, -1, (16, 8, 12, 15)], 0.2, False, True]
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 1)
            cpu_input2, npu_input2 = item[1], item[1]
            cpu_output = cpu_op_exec_fp16(cpu_input1, cpu_input2, item[2], item[3])
            npu_output = self.npu_op_extc(npu_input1, npu_input2, item[2], item[3])
            #self.assertRtolEqual(cpu_output, npu_output)
 

instantiate_device_type_tests(TestDropout, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()