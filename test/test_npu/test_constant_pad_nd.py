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


class TestConstantPadNd(TestCase):
    
    def op_exec_cpu(self, input1, pad_shape):
        output = torch.constant_pad_nd(input1, pad_shape)
        output = output.numpy()
        
        return output

    def op_exec_npu(self, input1, pad_shape):
        input1 = input1.to("npu")
        output = torch.constant_pad_nd(input1, pad_shape)
        output = output.to("cpu")
        output = output.numpy()
        return output
        
    def test_constant_pad_nd_shape_format(self, device):
        shape_format = [  
            [[np.float32, 3, (25, 32, 1, 1)], (1,1)],
            [[np.float32, 0, [25, 32, 11, 11]], (2,2,2,2)],
            [[np.float32, 0, [25, 3, 22, 22]],(2,2,2,2,20,20)],
            [[np.float16, 3, [25, 12, 7, 7]], (20,20,20,20)],
            [[np.float16, 0, [25, 3, 22, 22]], (20,20,20,20,5,5,5,5)],
            [[np.float16, 4, (2, 3, 3, 3)], (1,1,1,20,5,5,5,5)],
            [[np.float16, 4, [100, 20, 7, 7]], (0,0,0,0,0,0,0,0)],
            [[np.float16, 0, [2,3,4,5]], (1,0,1,0,1,0,1,0)],
            [[np.float16, 4, [2]],(0,1)],
            [[np.float16, 0, [20,20]],(0,1,0,2)],
            [[np.float16, 0, [20,20,20]],(1,1,1,1) ],
            [[np.float16, 3, [1,1,1,1]], (1,1)],
            [[np.float16, 3, [1]], (1,1)],
            [[np.float16, 0, [50, 24, 56, 56]], (100, 100, 100, 100, 100, 100, 100, 100)],
        ]

        for item in shape_format:
            input_cpu, input_npu = create_common_tensor(item[0], 1, 1)
            pad_shape = item[1]
            cpu_output = self.op_exec_cpu(input_cpu, pad_shape)
            npu_output = self.op_exec_npu(input_npu, pad_shape)
            
            
            self.assertRtolEqual(cpu_output, npu_output)
            


instantiate_device_type_tests(TestConstantPadNd, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
