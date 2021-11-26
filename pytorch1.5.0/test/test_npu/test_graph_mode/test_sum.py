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
import torch.nn as nn
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor
from graph_utils import RunFuncInGraphMode

class TestSum(TestCase):
    def cpu_op_exec(self, input1):
        output = input1.sum()
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = input1.sum()
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_dim(self, input1, dim, dtype):
        output = torch.sum(input1, dim, keepdim=True, dtype=dtype)
        output = output.numpy()
        return output

    def npu_op_exec_dim(self, input1, dim, dtype):
        output = torch.sum(input1, dim, keepdim=True, dtype=dtype)
        output = output.to("cpu")
        output = output.numpy()
        return output

    @RunFuncInGraphMode
    def test_sum_shape_format(self, device):
        shape_format = [
                [[np.float32, 0, [256]], [0]],
                [[np.float32, 0, [256, 1000]], [0]],
                [[np.int32, 0, [5, 256]], [0]],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 2, 100)
            cpu_output = self.cpu_op_exec_dim(cpu_input1, item[1], cpu_input1.dtype)
            npu_output = self.npu_op_exec_dim(npu_input1, item[1], cpu_input1.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestSum, globals(), except_for="cpu")
if __name__ == "__main__":
	run_tests()