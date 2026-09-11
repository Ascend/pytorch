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
Add validation cases for torch._utils and torch.utils APIs on NPU:
1. PyTorch community lacks sufficient direct validation for some APIs on NPU.
2. This file validates torch._utils._get_available_device_type and torch.utils._foreach_utils._has_foreach_support (extendable).
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils._foreach_utils import _has_foreach_support


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestTorchUtilsAPIs(TestCase):

    def test_get_available_device_type(self):
        self.assertEqual(torch._utils._get_available_device_type(), "npu")

    def test_has_foreach_support_for_npu_tensors(self):
        device = torch.device(device_type)
        tensors = [torch.ones(2, device=device), None]

        self.assertTrue(_has_foreach_support(tensors, device))
        self.assertFalse(_has_foreach_support([tensors[0], object()], device))


if __name__ == "__main__":
    run_tests()
