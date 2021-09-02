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


class TestExpand(TestCase):
    def cpu_op_exec(self, input1, size):
        output = input1.expand(size)
        output = output.numpy()
        return output

    def npu_op_exec(self,input1, size):
        output = input1.expand(size)
        output = output.cpu().numpy()
        return output

    def test_expand(self, device):
        shape_format = [
                [[np.float32, 0, [1, 3]], (3, 3)],
                [[np.float16, 0, [5, 1]], (-1, 7)],
                [[np.int32, 0, [1, 1]], (3, 3)],
    	]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1])
            npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestExpand, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
