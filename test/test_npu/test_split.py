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

# coding: utf-8

import torch
import numpy as np
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class testSplit(TestCase):

    def cpu_op_exec(self, input1, split_size, dim):
        output_tuple = torch.split(input1,split_size, dim=dim)
        listtuple1 = []
        for i in range(len(output_tuple)):
            listtuple1 += list(output_tuple[i].contiguous().view(-1))
        output = torch.tensor(listtuple1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, split_size, dim):
        output_tuple = torch.split(input1, split_size, dim=dim)
        listtuple1 = []
        for i in range(len(output_tuple)):
            listtuple1 += list(output_tuple[i].contiguous().view(-1))
        output = torch.tensor(listtuple1)
        output = output.to("cpu")
        output = output.numpy()
        return output
    
    def test_split_common_shape_format(self, device):
        shape_format = [
                [[np.float32, 0 , (1, 4, 2, 3)], 3, 1],
                [[np.float32, 0, (8,4)], [1,2,1,2,2],0],
                [[np.float16, 0 , (1, 4, 2, 3)], 3, 1],
                [[np.float16, 0, (8,4)], [1,2,1,2,2],0],
                [[np.int32, 0 , (1, 4, 2, 3)], 3, 1],
                [[np.int32, 0, (8,4)], [1,2,1,2,2],0],
                [[np.int64, 0 , (1, 4, 2, 3)], 3, 1],
                [[np.int64, 0, (8,4)], [1,2,1,2,2],0],
                [[np.double, 0 , (1, 4, 2, 3)], 3, 1],
                [[np.double, 0, (8,4)], [1,2,1,2,2],0]
                ]
        for item in shape_format:            
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 10)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1], item[2])
            npu_output = self.npu_op_exec(npu_input1, item[1], item[2])
            self.assertRtolEqual(cpu_output, npu_output)       
    
instantiate_device_type_tests(testSplit, globals(), except_for="cpu")
if __name__ == "__main__":
    torch.npu.set_device("npu:6")
    run_tests()
