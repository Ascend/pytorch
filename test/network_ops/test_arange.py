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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestArange(TestCase):
    def test_arange_default(self):
        shape_format = [
            [0, 100, 2],
            [1, 100, 1],
            [5, 100, 3],
        ]

        for item in shape_format:
            cpu_output_default = torch.arange(item[0], item[1], item[2],
                                              device="cpu").numpy()
            npu_output_default = torch.arange(item[0], item[1], item[2],
                                              device="npu").cpu().numpy()
            self.assertRtolEqual(cpu_output_default, npu_output_default)

    def test_arange(self):
        shape_format = [
            [0, 100, 2, torch.float32],
            [1, 100, 1, torch.int32],
            [5, 100, 3, torch.int64],
        ]

        for item in shape_format:
            cpu_output = torch.arange(item[0], item[1], item[2], dtype=item[3],
                                      device="cpu").numpy()
            npu_output = torch.arange(item[0], item[1], item[2], dtype=item[3],
                                      device="npu").cpu().numpy()
            self.assertRtolEqual(cpu_output, npu_output)

    def test_arange_out(self):
        shape_format = [
            [0, 100, 1, torch.float32, [np.float32, 0, [10]]],
            [1, 100, 2, torch.int32, [np.int32, 0, [20]]],
            [5, 100, 3, torch.int64, [np.int64, 0, [30]]],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[4], 0, 10)
            cpu_output = torch.arange(item[0], item[1], item[2],
                                      dtype=item[3], device="cpu").numpy()
            npu_output = torch.arange(item[0], item[1], item[2], out=npu_input1,
                                      dtype=item[3], device="npu").cpu().numpy()
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
