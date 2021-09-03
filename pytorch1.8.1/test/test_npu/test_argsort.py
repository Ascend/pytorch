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

class TestArgsort(TestCase):
    def cpu_op_exec(self, input, dim, descending):
        output = torch.argsort(input, dim, descending) 
        output = output.numpy()
        output = output.astype("int32")
        return output

    def cpu_fp16_op_exec(self, input, dim, descending):
        input = input.to(torch.float32)
        output = torch.argsort(input, dim, descending) 
        output = output.numpy()
        output = output.astype("int32")
        return output
    
    def npu_op_exec(self, input, dim, descending):
        output = torch.argsort(input, dim, descending) 
        output = output.to("cpu")
        output = output.numpy()
        output = output.astype("int32")
        return output
            
    def test_argsort_shape_format_fp32(self, device):
        shape_format = [
                [[np.float32, -1, (1, 12, 5, 8)], -1, False], 
                [[np.float32, -1, (2, 3, 13)], 2, True], 
                [[np.float32, -1, (5, 20)], 1, False], 
                [[np.float32, -1, (1,)], 0, False]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], -100, 100)
            cpu_output = self.cpu_op_exec(cpu_input, item[1], item[2])
            npu_output = self.npu_op_exec(npu_input, item[1], item[2])
            self.assertRtolEqual(cpu_output, npu_output)
    
    def test_argsort_shape_format_fp16(self, device):
        shape_format = [
                #[[np.float16, -1, (2, 31, 15, 7)], -2, False], 
                [[np.float16, -1, (2, 5, 23)], 1, False], 
                [[np.float16, -1, (5, 12)], -1, True], 
                [[np.float16, -1, (1, 1)], 0, False]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], -100, 100)
            cpu_output = self.cpu_fp16_op_exec(cpu_input, item[1], item[2])
            npu_output = self.npu_op_exec(npu_input, item[1], item[2])
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestArgsort, globals(), except_for="cpu") 
if __name__ == "__main__":
    torch.npu.set_device("npu:5")
    run_tests()
