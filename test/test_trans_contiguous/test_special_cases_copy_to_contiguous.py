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

import os
import torch
import torch_npu
import numpy as np

from torch_npu.testing.testcase import TestCase, run_tests

os.environ["COMBINED_ENABLE"] = "1"  # Open combined-view cases optimization


class TestSpecialCasesCopyToContiguous(TestCase):
    def test_expand_copy_to_slice_tensor(self, device="npu"):
        cpu_input = torch.zeros((2, 10)).bool()
        cpu_out = cpu_input
        cpu_out[0, :3] = True
        npu_out = cpu_input.npu()
        npu_out[0, :3] = True
        self.assertRtolEqual(npu_out.to("cpu").numpy(), cpu_out.numpy())
           

if __name__ == "__main__":
    run_tests()