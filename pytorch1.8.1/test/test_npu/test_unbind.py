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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestUnbind(TestCase):

    def cpu_op_exec(self, input1, dim):
        output_tuple= torch.unbind(input1, dim=dim)
        listtuple1 = []
        for i in range(len(output_tuple)):
            listtuple1 += list(output_tuple[i].contiguous().view(-1))
        output = torch.tensor(listtuple1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, dim):
        output_tuple = torch.unbind(input1, dim=dim)
        listtuple1 = []
        for i in range(len(output_tuple)):
            listtuple1 += list(output_tuple[i].contiguous().view(-1))
        output = torch.tensor(listtuple1)
        output = output.to("cpu")
        output = output.numpy()
        return output
    
    def test_unbind_common_shape_format(self, device):
        shape_format = [
                [[np.float32, 0 , (1, 4, 2, 3)], 1],
                [[np.float32, 0, (1, 3, 2, 3)], 2],
                [[np.float32, 0, (3, 2, 3)], 2],
                [[np.float32, 0, ( 2, 3)], 0],
                [[np.float16, 0 , (1, 4, 2, 3)], 1],
                [[np.float16, 0, (1, 3, 2, 3)], 3],
                [[np.float16, 0, (3, 2, 3)], 2],
                [[np.float16, 0, ( 2, 3)], 0],
                [[np.int32, 0 , (1, 4, 2, 3)], 1],
                [[np.int32, 0, (1, 3, 2, 3)], 3],
                [[np.int32, 0, (3, 2, 3)], 2],
                [[np.int32, 0, ( 2, 3)], 0],
                [[np.int16, 0 , (1, 4, 2, 3)], 1],
                [[np.int16, 0, (1, 3, 2, 3)], 3],
                [[np.int16, 0, (3, 2, 3)], 2],
                [[np.int16, 0, ( 2, 3)], 0],
                [[np.int8, 0 , (1, 4, 2, 3)], 1],
                [[np.int8, 0, (1, 3, 2, 3)], 3],
                [[np.int8, 0, (3, 2, 3)], 2],
                [[np.int8, 0, ( 2, 3)], 0],
                [[np.uint8, 0 , (1, 4, 2, 3)], 1],
                [[np.uint8, 0, (1, 3, 2, 3)], 3],
                [[np.uint8, 0, (3, 2, 3)], 2],
                [[np.uint8, 0, ( 2, 3)], 0],
                [[np.int64, 0 , (1, 4, 2, 3)], 1],
                [[np.int64, 0, (1, 3, 2, 3)], 3],
                [[np.int64, 0, (3, 2, 3)], 2],
                [[np.int64, 0, ( 2, 3)], 0]
                ]
        for item in shape_format:            
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1])
            npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)       
    
instantiate_device_type_tests(TestUnbind, globals(), except_for="cpu")
if __name__ == "__main__":
    torch.npu.set_device("npu:6")
    run_tests()
