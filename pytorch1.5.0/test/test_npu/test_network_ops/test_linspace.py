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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestLinspace(TestCase):
    def test_linspace(self, device):
        shape_format = [
            [0, 100, 10, torch.float32],
            [1, 100, 20, torch.int32],
        ]

        for item in shape_format:
            cpu_output = torch.linspace(item[0], item[1], item[2], dtype=item[3],
                device="cpu").numpy()
            npu_output = torch.linspace(item[0], item[1], item[2], dtype=item[3],
                device="npu").cpu().numpy()
            self.assertRtolEqual(cpu_output, npu_output)

    def test_linspace_out(self, device):
        shape_format = [
            [0, 100, 10, torch.float32, [np.float32, 0, [10]]],
            [1, 100, 20, torch.int32, [np.int32, 0, [20]]],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[4], 0, 10)
            cpu_output = torch.linspace(item[0], item[1], item[2], out = cpu_input1,
                dtype=item[3], device="cpu").numpy()
            npu_output = torch.linspace(item[0], item[1], item[2], out = npu_input1,
                dtype=item[3], device="npu").cpu().numpy()
            self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestLinspace, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
