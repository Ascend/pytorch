# Copyright (c) 2020 Huawei Technologies Co., Ltd
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

import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.contrib.function import npu_fast_condition_index_put


class TestIndexOp(TestCase):
    def npu_slow_index_op_exec(self, input1):
        condition = input1 < 0.5
        value = 0.
        input1[condition] = value
        return input1

    def npu_fast_index_op_exec(self, input1):
        condition = input1 < 0.5
        value = 0.
        return npu_fast_condition_index_put(input1, condition, value)

    def test_npu_index_op(self):
        dtype_list = [np.float16, np.float32]
        format_list = [-1, 0, 2]
        shape_list = [
            [2, 3, 7, 7],
            [1, 2, 3, 6],
            [6, 5, 8, 10],
            [2, 5, 6, 8]
        ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 10)
            npu_slow_output = self.npu_slow_index_op_exec(npu_input)
            npu_fast_output = self.npu_fast_index_op_exec(npu_input)
            self.assertRtolEqual(npu_slow_output.cpu(), npu_fast_output.cpu())


if __name__ == "__main__":
    run_tests()
