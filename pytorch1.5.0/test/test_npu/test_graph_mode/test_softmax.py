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
from graph_utils import graph_mode

class TestSoftmax(TestCase):
    def cpu_op_exec(self, input_data, dim):
        m = torch.nn.Softmax(dim) 
        output = m(input_data)
        output = output.numpy()
        return output
    
    def npu_op_exec(self, input_data, dim):
        m = torch.nn.Softmax(dim) 
        output = m(input_data).to("cpu")
        output = output.numpy()
        return output
            
    @graph_mode     
    def test_softmax_shape_format_fp32(self, device):
        shape_format = [
                [[np.float32, 0, (1, 12, 5, 8)], 0], 
                [[np.float32, 0, (2, 31, 53)], 0], 
                [[np.float32, 0, (5, 20)], 0], 
                [[np.float32, 0, (1)], 0]     
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], -2, 2)
            dim = item[1]
            cpu_output = self.cpu_op_exec(cpu_input, item[1])
            npu_output = self.npu_op_exec(npu_input, item[1])
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestSoftmax, globals(), except_for="cpu") 
if __name__ == "__main__":
    run_tests()
