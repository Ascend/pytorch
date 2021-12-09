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
import torch.nn as nn
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestNpuPad(TestCase):
    def test_npu_pad(self, device):
        npu_input = torch.ones(2, 2).npu()
        pads = (1, 1, 1, 1)
        benchmark = torch.tensor([[0., 0., 0., 0.],
                                  [0., 1., 1., 0.],
                                  [0., 1., 1., 0.],
                                  [0., 0., 0., 0.]])
        npu_output = torch.npu_pad(npu_input, pads)
        npu_output = npu_output.cpu().detach()
        self.assertRtolEqual(benchmark, npu_output)

instantiate_device_type_tests(TestNpuPad, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
