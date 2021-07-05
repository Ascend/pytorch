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

class TestNarrowCopy(TestCase):
    def cpu_op_exec(self, data, dim, start, length):
        output = data.narrow_copy(dim, start, length)
        output = output.to("cpu")
        output = output.detach().numpy().astype(np.int32)
        return output
 
    def npu_op_exec(self, data, dim, start, length):
        output = data.narrow_copy(dim, start, length)
        output = output.to("cpu")
        output = output.detach().numpy().astype(np.int32)
        return output
    
    def test_narrow_copy_1(self, device):
        data = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        data_npu = data.to("npu")
        
        cpu_output = self.cpu_op_exec(data, 0, 0, 2)
        npu_output = self.npu_op_exec(data_npu, 0, 0, 2)

        self.assertRtolEqual(cpu_output, npu_output)
    
    def test_narrow_copy_2(self, device):
        data = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        data_npu = data.to("npu")
        
        cpu_output = self.cpu_op_exec(data, 1, 1, 1)
        npu_output = self.npu_op_exec(data_npu, 1, 1, 1)

        self.assertRtolEqual(cpu_output, npu_output)
    
    def test_narrow_copy_3(self, device):
        data = torch.tensor([[[16,5,7,4],[16,5,7,4]],[[16,5,7,4],[16,5,7,4]],[[16,5,7,4],[16,5,7,4]]])
        data_npu = data.to("npu")
        cpu_output = self.cpu_op_exec(data, 2, -2, 1)
        npu_output = self.npu_op_exec(data_npu, 2, -2, 1)
        
        self.assertRtolEqual(cpu_output, npu_output)
    
    def test_narrow_copy_4(self, device):
        data = torch.tensor([[[16,5,7,4],[16,5,7,4]],[[16,5,7,4],[16,5,7,4]],[[16,5,7,4],[16,5,7,4]]])
        data_npu = data.to("npu")
        cpu_output = self.cpu_op_exec(data, -1, -2, 1)
        npu_output = self.npu_op_exec(data_npu, -1, -2, 1)
        
        self.assertRtolEqual(cpu_output, npu_output)
    
instantiate_device_type_tests(TestNarrowCopy, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
