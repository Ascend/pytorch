# Copyright (c) 2020 Huawei Technologies Co., Ltd
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
import copy
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestAny(TestCase):
    def cpu_op_exec(self, input1, dim, keepdim):
        output = input1.any(dim=dim, keepdim=keepdim)
        return output.numpy()

    def npu_op_exec(self, input1, dim, keepdim):
        output = input1.any(dim=dim, keepdim=keepdim)
        output = output.to("cpu")
        return output.numpy()

    def cpu_op_dim_exec(self, input1, dim):
        output = input1.any(dim)
        return output.numpy()

    def npu_op_dim_exec(self, input1, dim):
        output = input1.any(dim)
        output = output.to("cpu")
        return output.numpy()

    def test_any_shape_format(self, device):
        shape_format = [
                [[np.bool, 0, 1], 0, True],
                [[np.bool, 0, (64, 10)], 1, False],
                [[np.bool, 0, (1,3,4)], 2, True],
                [[np.bool, 0, (3,4,6,7)], 2, False],
                [[np.bool, 30, (1,3,4,6,7)], 4, True]
                ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            npu_input = copy.deepcopy(cpu_input).npu()
            cpu_output = self.cpu_op_exec(cpu_input, item[1], item[2])
            npu_output = self.npu_op_exec(npu_input, item[1], item[2])
            self.assertRtolEqual(cpu_output, npu_output)

            cpu_dim_output = self.cpu_op_dim_exec(cpu_input, item[1])
            npu_dim_output = self.npu_op_dim_exec(npu_input, item[1])
            self.assertRtolEqual(cpu_dim_output, npu_dim_output)
    
instantiate_device_type_tests(TestAny, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
