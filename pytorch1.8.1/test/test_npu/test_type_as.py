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


class TestTypeAs(TestCase):        
    def cpu_op_exec(self, input1, input2):
        tensor1 = input1
        tensor2 = input2
        output = tensor1.type_as(tensor2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        tensor1 = input1
        tensor2 = input2
        output = tensor1.type_as(tensor2)
        output = output.to("cpu")
        output = output.numpy()
        return output 

    def test_type_as_int32_shape_format(self, device):
        shape_format = [
                [[np.float32, -1, (4, 3)],    [np.int32, -1, (4, 3)]],
                [[np.float32, -1, (4, 3, 1)], [np.int32, -1, (4, 3, 1)]],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)   

    def test_type_as_float32_shape_format(self, device):
        shape_format = [
                [[np.int32, -1, (8, 5)],    [np.float32, -1, (8, 5)]],
                [[np.int32, -1, (9, 4, 2)], [np.float32, -1, (9, 4, 2)]],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)  

   
instantiate_device_type_tests(TestTypeAs, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:5")
    run_tests()
