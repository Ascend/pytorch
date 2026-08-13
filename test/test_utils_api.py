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
Add validation cases for torch._utils APIs on NPU:
1. PyTorch community lacks sufficient direct validation for some APIs on NPU.
2. This file validates torch._utils._get_available_device_type (extendable).
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests


class TestTorchUtilsAPIs(TestCase):

    def test_get_available_device_type(self):
        self.assertEqual(torch._utils._get_available_device_type(), "npu")


if __name__ == "__main__":
    run_tests()
