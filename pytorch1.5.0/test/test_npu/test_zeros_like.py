# Copyright (c) 2020, Huawei Technologies.
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
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestZerosLike(TestCase):
    def cpu_op_exec(self, input1):
        output = torch.zeros_like(input1)
        return output

    def npu_op_exec(self, input1):
        output = torch.zeros_like(input1) 
        output = output.to("cpu")
        return output

    def cpu_op_dtype_exec(self, input1, dtype):
        output = torch.zeros_like(input1, dtype=dtype)
        return output

    def npu_op_dtype_exec(self, input1, dtype):
        output = torch.zeros_like(input1, dtype=dtype)
        output = output.to("cpu")
        return output
 
    def test_zeros_like_shape_format(self, device):
        shape_format = [
               [ [np.float32, 0, (1, 6, 4)] ],
               [ [np.float32, 3, (2, 4, 5)] ]
               ]
        for item in shape_format:            
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_zeros_like_dtype_shape_format(self, device):
        shape_format = [
               [ [np.float32, 0, (1, 6, 4)], torch.float32],
               [ [np.float32, 3, (2, 4, 5)], torch.float16 ],
               ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_dtype_exec(cpu_input, item[1])
            npu_output = self.npu_op_dtype_exec(npu_input, item[1])
            self.assertRtolEqual(cpu_output, npu_output)
        
instantiate_device_type_tests(TestZerosLike, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
