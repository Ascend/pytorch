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


class TestLinspace(TestCase):
    def test_linspace(self, device="npu"):
        shape_format = [
            [0, 100, 10, torch.float32,
             torch.tensor([0., 11.111111, 22.222221, 33.333332, 44.444443,
                          55.555557, 66.666664, 77.77778, 88.888885, 100.])],
            [1, 100, 20, torch.int32,
             torch.tensor([1, 6, 11, 16, 21, 27, 32, 37, 42,
                          47, 53, 58, 63, 68, 73, 79, 84, 89, 94, 100], dtype=torch.int32)],
        ]

        for item in shape_format:
            cpu_output = torch.linspace(item[0], item[1], item[2], dtype=item[3],
                                        device="cpu")
            npu_output = torch.linspace(item[0], item[1], item[2], dtype=item[3],
                                        device="npu").cpu()
            benchmark15 = item[4]
            self.assertRtolEqual(benchmark15, npu_output)

    def test_linspace_out(self, device="npu"):
        shape_format = [
            [0, 100, 10, torch.float32, [np.float32, 0, [10]],
             torch.tensor([0., 11.111111, 22.222221, 33.333332, 44.444443,
                          55.555557, 66.666664, 77.77778, 88.888885, 100.])],
            [1, 100, 20, torch.int32, [np.int32, 0, [20]],
             torch.tensor([1, 6, 11, 16, 21, 27, 32, 37, 42,
                          47, 53, 58, 63, 68, 73, 79, 84, 89, 94, 100], dtype=torch.int32)],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[4], 0, 10)
            cpu_output = torch.linspace(item[0], item[1], item[2], out=cpu_input1,
                                        dtype=item[3], device="cpu")
            npu_output = torch.linspace(item[0], item[1], item[2], out=npu_input1,
                                        dtype=item[3], device="npu").cpu()
            benchmark15 = item[5]
            self.assertRtolEqual(benchmark15, npu_output)


if __name__ == "__main__":
    run_tests()
