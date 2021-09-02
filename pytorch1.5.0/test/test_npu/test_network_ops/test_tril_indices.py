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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestTrilIndices(TestCase):
    def cpu_op_exec(self, r, c):
        output = torch.tril_indices(r, c, device="cpu")
        output = output.numpy()
        return output

    def npu_op_exec(self, r, c):
        output = torch.tril_indices(r, c, device="npu")
        output = output.to("cpu")
        output = output.numpy()
        return output
        
    def test_tril_indices(self, device):
        shape_format = [
            [3, 3],
            [4, 3, -1],
            [4, 3, 1],
        ]
        for item in shape_format:
            cpu_output = self.cpu_op_exec(item[0], item[1])
            npu_output = self.npu_op_exec(item[0], item[1])
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestTrilIndices, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
