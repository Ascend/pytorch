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


class TestIsFloatingPiont(TestCase):
    def test_is_floating_point(self, device):
        shape_format = [
            [0.36, torch.float32],
            [1, torch.int32],
            [1, torch.float32],
            [1, torch.float16],
            [1, torch.int8],
        ]

        for item in shape_format:
            inputs = torch.tensor([item[0]], dtype=item[1])
            cpu_out = inputs.is_floating_point()
            cpu_out = np.array(cpu_out, dtype=np.int32)
            inputs = inputs.to("npu")
            npu_out = inputs.is_floating_point()
            npu_out = np.array(npu_out, dtype=np.int32)
            self.assertRtolEqual(cpu_out, npu_out)


instantiate_device_type_tests(TestIsFloatingPiont, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
