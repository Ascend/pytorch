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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestPermute(TestCase):
    def test_permute_1(self, device):
        inputs = torch.randn(2, 3, 5)
        cpu_out = inputs.permute(2, 0, 1)
        inputs = inputs.to("npu")
        npu_out = inputs.permute(2, 0, 1)
        self.assertRtolEqual(cpu_out.detach().numpy(), npu_out.cpu().detach().numpy())

    def test_permute_2(self, device):
        inputs = torch.randn(2, 5, 6, 9)
        cpu_out = inputs.permute(2, 3, 0, 1)
        inputs = inputs.to("npu")
        npu_out = inputs.permute(2, 3, 0, 1)
        self.assertRtolEqual(cpu_out.detach().numpy(), npu_out.cpu().detach().numpy())

    def test_permute_3(self, device):
        inputs = torch.randn(2, 4, 6, 8, 10)
        cpu_out = inputs.permute(2, 0, 1, 4, 3)
        inputs = inputs.to("npu")
        npu_out = inputs.permute(2, 0, 1, 4, 3)
        self.assertRtolEqual(cpu_out.detach().numpy(), npu_out.cpu().detach().numpy())

instantiate_device_type_tests(TestPermute, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:6")
    run_tests()
