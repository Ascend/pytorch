#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
Add validation cases for torch._check_is_size on NPU:
1. PyTorch community lacks direct validation for torch._check_is_size, so this file is added.
2. This file validates:
   torch._check_is_size
(extendable)
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestCheckIsSize(TestCase):

    def test_passes_for_non_negative_int(self):
        torch._check_is_size(0)
        torch._check_is_size(1)
        torch._check_is_size(100)

    def test_raises_for_negative_int(self):
        with self.assertRaises(RuntimeError):
            torch._check_is_size(-1)

    def test_passes_within_max_bound(self):
        torch._check_is_size(5, max=10)

    def test_fails_above_max_bound(self):
        with self.assertRaises(RuntimeError):
            torch._check_is_size(11, max=10)

    def test_passes_at_max_equal_i(self):
        """Boundary: i == max is a valid size (0 <= i <= max)."""
        torch._check_is_size(10, max=10)

    def test_passes_with_explicit_max_none(self):
        torch._check_is_size(5, max=None)

    def test_message_omitted(self):
        """Default message=None: no custom message, just the default."""
        torch._check_is_size(0)

    def test_message_explicit_none(self):
        """Explicit message=None should behave the same as omitted."""
        torch._check_is_size(0, message=None)

    def test_message_callable_produces_custom_error(self):
        """message passed as a callable; on failure it produces the custom message."""
        with self.assertRaises(RuntimeError) as cm:
            torch._check_is_size(-1, message=lambda: "custom error msg")
        self.assertIn("custom error msg", str(cm.exception))

    def test_message_format_string(self):
        """message must be a callable; passing a non-callable raises TypeError."""
        with self.assertRaises(TypeError):
            torch._check_is_size(-1, message="size must be non-negative")

    def test_raises_for_non_int_type(self):
        with self.assertRaises(TypeError):
            torch._check_is_size("not_an_int")

    def test_raises_for_invalid_max_type(self):
        with self.assertRaises(TypeError):
            torch._check_is_size(5, max="invalid_max")

    def test_check_tensor_size_value(self):
        x = torch.zeros(3, 4, device=device_type)
        torch._check_is_size(x.size(0))
        torch._check_is_size(x.size(1), max=8)


if __name__ == "__main__":
    run_tests()
