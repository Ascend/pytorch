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
import torch.nn as nn
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestPermute(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = input1.permute(input2);
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        input1 = input1.to("npu")
        output = input1.permute(input2);
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_permute(self, device):
        shape_format = [
            [[2, 3, 5], (2, 0, 1), torch.float32],
            [[2, 5, 6, 9], (2, 0, 3, 1), torch.float32],
            [[2, 4, 6, 8, 10], (2, 3, 4, 0, 1), torch.float32],
        ]
        for item in shape_format:
            cpu_input1 =  torch.randn(item[0], dtype=item[2])
            cpu_output = self.cpu_op_exec(cpu_input1, item[1])
            npu_output = self.npu_op_exec(cpu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestPermute, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
