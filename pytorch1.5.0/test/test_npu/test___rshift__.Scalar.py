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
import numpy as np
import sys
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestRshiftScalar(TestCase):

    def cpu_op_exec(self, input, other):
        output = input.__rshift__(other)
        return output.numpy()

    def npu_op_exec(self, input, other):
        output = input.__rshift__(other).npu()
        return output.cpu().numpy()


    def test_cast_Char_common_shape_format(self, device):
        shape_format = [
            [[np.int64, -1, (4, 3)]],
            [[np.int32, -1, (4, 3, 1)]],
            [[np.int8, -1, (2, 3)]],
            [[np.float32, -1, (4, 3, 1)]],
            [[np.float16, -1, (4, 3, 1)]],
            [[np.uint8, -1, (4, 3, 1)]]
        ]
        other_list = [0, 1, -1, 1.5, -1.5, 10, -10, 100, -100, 1000000, -1000000]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], -100, 100)
            cpu_input = cpu_input.to(torch.float32)
            for other in other_list:
                cpu_output = self.cpu_op_exec(cpu_input, other)
                npu_output = self.npu_op_exec(npu_input, other)
                cpu_output = cpu_output.astype(np.float16)
                self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestRshiftScalar, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:3")
    run_tests()