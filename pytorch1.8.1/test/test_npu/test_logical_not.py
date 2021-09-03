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

class TestLogicalNot(TestCase):
    def cpu_op_exec(self, input):
        output = torch.logical_not(input)
        output = output.numpy()
        return output

    def npu_op_exec(self, input):
        output = torch.logical_not(input)      
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_logical_not_common_shape_format(self, device):
        shape_format = [
                [[np.int8, -1, 1]],
                [[np.int8, -1, (64, 10)]],
                [[np.int8, -1, (256, 2048, 7, 7)]],
                [[np.int8, -1, (32, 1, 3, 3)]],
                [[np.int32, -1, (64, 10)]],
                [[np.int32, -1, (256, 2048, 7, 7)]],
                [[np.int32, -1, (32, 1, 3, 3)]],
                [[np.uint8, -1, (64, 10)]],
                [[np.uint8, -1, (256, 2048, 7, 7)]],
                [[np.uint8, -1, (32, 1, 3, 3)]],
                [[np.float16, -1, (64, 10)]],
                [[np.float16, -1, (256, 2048, 7, 7)]],
                [[np.float16, -1, (32, 1, 3, 3)]],
                [[np.float32, -1, (64, 10)]],
                [[np.float32, -1, (256, 2048, 7, 7)]],
                [[np.float32, -1, (32, 1, 3, 3)]],
                [[np.bool, -1, (64, 10)]],
                [[np.bool, -1, (256, 2048, 7, 7)]]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 10)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)  



instantiate_device_type_tests(TestLogicalNot, globals(), except_for="cpu")
if __name__ == "__main__":
    torch.npu.set_device("npu:5")
    run_tests()
        
