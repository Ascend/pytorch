# Copyright (c) 2026 Huawei Technologies Co., Ltd
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

"""
Add validation cases for torch.utils._foreach_utils APIs on NPU:
1. PyTorch community lacks sufficient and direct API validations for some APIs,
   so this file is added.
2. This file validates _device_has_foreach_support (extendable).
"""

from unittest.mock import patch

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils._foreach_utils import _device_has_foreach_support

device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
)


class TestForeachUtils(TestCase):
    """Test foreach device support detection."""

    def test_supported_device_types(self):
        supported_device_types = (
            "cpu",
            torch._C._get_privateuse1_backend_name(),
        )

        for supported_device_type in supported_device_types:
            with self.subTest(device_type=supported_device_type):
                self.assertTrue(
                    _device_has_foreach_support(torch.device(supported_device_type))
                )

    def test_current_accelerator_and_device_index(self):
        self.assertTrue(_device_has_foreach_support(torch.device(device_type)))
        self.assertTrue(
            _device_has_foreach_support(device=torch.device(f"{device_type}:0"))
        )

    def test_unsupported_device_types(self):
        for unsupported_device_type in ("meta", "mps"):
            with self.subTest(device_type=unsupported_device_type):
                self.assertFalse(
                    _device_has_foreach_support(torch.device(unsupported_device_type))
                )

    def test_torchscript_disables_foreach_support(self):
        supported_device_types = (
            "cpu",
            torch._C._get_privateuse1_backend_name(),
        )

        with patch("torch.jit.is_scripting", return_value=True):
            for supported_device_type in supported_device_types:
                with self.subTest(device_type=supported_device_type):
                    self.assertFalse(
                        _device_has_foreach_support(
                            torch.device(supported_device_type)
                        )
                    )

    def test_invalid_device_types(self):
        for invalid_device in (None, "npu", 0, ["npu"]):
            with self.subTest(device=invalid_device):
                with self.assertRaises(AttributeError):
                    _device_has_foreach_support(invalid_device)

    def test_invalid_argument_count(self):
        with self.assertRaises(TypeError):
            _device_has_foreach_support()

        with self.assertRaises(TypeError):
            _device_has_foreach_support(torch.device(device_type), None)


if __name__ == "__main__":
    run_tests()
