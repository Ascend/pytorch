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
from util_test import create_common_tensor

class TestBincount(TestCase):
    def cpu_op_exec(self, input, weights):
        output = torch.bincount(input,weights) 
        output = output.numpy()
        return output
 
    def npu_op_exec(self, input, weights):
        output = torch.bincount(input,weights) 
        output = output.to("cpu")
        output = output.numpy()
        return output  
       
    def test_bincount_common_shape_format(self, device):
        shape_format = [
                [[np.int16, -1, (1,)], 0],
                [[np.int16, -1, (18,)], 1],
                [[np.int16, -1, (32,), 2]],
                [[np.int16, -1, (100,), 3]],
                [[np.int32, -1, (10,)], 0],
                [[np.int32, -1, (8,)], 1],
                [[np.int32, -1, (32,), 2]],
                [[np.int32, -1, (124,), 3]],
                [[np.int64, -1, (1,)], 0],
                [[np.int64, -1, (8,)], 1],
                [[np.int64, -1, (32,), 2]],
                [[np.int64, -1, (100,), 3]],
                [[np.uint8, -1, (11,)], 0],
                [[np.uint8, -1, (80,)], 1],
                [[np.uint8, -1, (320,), 2]],
                [[np.uint8, -1, (1024,), 3]],
                [[np.uint8, -1, (11,)], 0],
                [[np.uint8, -1, (18,)], 1],
                [[np.uint8, -1, (32,), 2]],
                [[np.uint8, -1, (100,), 3]],

        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], -1, 1)
            cpu_weights, npu_weights = create_common_tensor(item[0], -1, 1)
            cpu_output = self.cpu_op_exec(cpu_input, cpu_weights)
            npu_output = self.npu_op_exec(npu_input, npu_weights)
            self.assertRtolEqual(cpu_output, npu_output)
            
instantiate_device_type_tests(TestBincount, globals(), except_for="cpu")
if __name__ == "__main__":
    torch.npu.set_device("npu:6")
    run_tests()
