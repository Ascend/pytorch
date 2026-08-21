# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
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

"""
Add validation cases for PyTorch extension utility APIs on NPU:
1. Some PyTorch community cases depend on renaming the PrivateUse1 backend.
2. This file directly validates torch._utils._get_device_index (extendable).
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests


class TestExtensionUtilsAPIs(TestCase):

    def test_get_device_index(self):
        current_device = torch.npu.current_device()

        self.assertEqual(torch._utils._get_device_index("npu:0"), 0)
        self.assertEqual(torch._utils._get_device_index(torch.device("npu:0")), 0)
        self.assertEqual(torch._utils._get_device_index(0), 0)

        for device in (None, torch.device("npu")):
            with self.assertRaises(ValueError):
                torch._utils._get_device_index(device)
            self.assertEqual(
                torch._utils._get_device_index(device, optional=True), current_device
            )

        cpu_device = torch.device("cpu")
        with self.assertRaises(ValueError):
            torch._utils._get_device_index(cpu_device)
        self.assertEqual(torch._utils._get_device_index(cpu_device, allow_cpu=True), -1)

        with self.assertRaises(ValueError):
            torch._utils._get_device_index(object())


if __name__ == "__main__":
    run_tests()
