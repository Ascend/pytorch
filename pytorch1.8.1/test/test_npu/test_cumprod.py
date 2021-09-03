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

class TestCumprod(TestCase):

    def cpu_op_exec(self,input1, dim):
        output = torch.cumprod(input1, dim)
        output = output.numpy()
        return output

    def npu_op_exec(self,input1, dim):
        output = torch.cumprod(input1, dim)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self,input1, input2, dim):
        torch.cumprod(input1, dim, out=input2)
        output = input2.to("cpu")
        output = output.numpy()
        return output

    def test_cumprod_common_shape_format(self, device):
        shape_format = [
                [[np.float32, 0, (5,3)]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 10)
            dim = 0
            cpu_output = self.cpu_op_exec(cpu_input1, dim)
            npu_output = self.npu_op_exec(npu_input1, dim)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_cumprod_out_common_shape_format(self, device):
        shape_format = [
                [[np.float32, 0, (4,3)],    [np.float32, 0, (4,3)]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 1, 10)
            dim = 0
            cpu_output = self.cpu_op_exec(cpu_input1, dim)
            npu_output = self.npu_op_exec_out(npu_input1, npu_input2, dim)
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestCumprod, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
