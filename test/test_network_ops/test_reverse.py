# Copyright (c) 2022, Huawei Technologies.All rights reserved.
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


class TestReverse(TestCase):
    def test_reverse(self, device="npu"):
        cpu_input = np.random.uniform(0, 255, (1, 3, 224, 224)).astype(np.uint8)
        npu_input = torch.from_numpy(cpu_input).npu()
        cpu_output = np.flip(cpu_input, 0)
        npu_output = torch_npu.reverse(npu_input, [0]).cpu().numpy()
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()