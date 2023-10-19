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


class TestDefault(TestCase):
    def test_isnan(self, device="npu"):
        cpu_input = torch.arange(1., 10)
        npu_input = cpu_input.npu()

        cpu_output = torch.isnan(cpu_input)
        npu_output = torch.isnan(npu_input)
        self.assertRtolEqual(cpu_output, npu_output.cpu())

    def test_unfold(self, device="npu"):
        cpu_input = torch.arange(1., 8)
        npu_input = cpu_input.npu()

        cpu_output = cpu_input.unfold(0, 2, 1)
        npu_output = npu_input.unfold(0, 2, 1)
        self.assertRtolEqual(cpu_output, npu_output.cpu())


if __name__ == "__main__":
    run_tests()
