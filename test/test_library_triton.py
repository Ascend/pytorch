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

# Owner(s): ["module: library"]

"""
Add validation cases for torch._library.triton APIs on NPU:
1. PyTorch community lacks sufficient and direct API validations for
   torch._library.triton.set_wrap_triton_enabled and
   torch._library.triton.is_wrap_triton_enabled, so this file is added.
2. This file validates torch._library.triton.set_wrap_triton_enabled and
   torch._library.triton.is_wrap_triton_enabled (extendable).
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._triton import has_triton_package


class TestLibraryTriton(TestCase):
    """Test torch._library.triton.set_wrap_triton_enabled and is_wrap_triton_enabled."""

    def test_set_wrap_triton_enabled_default_is_true(self):
        """By default wrap_triton dispatch via HOP is enabled."""
        self.assertTrue(torch._library.triton.is_wrap_triton_enabled())

    def test_set_wrap_triton_enabled_disables_wrapping(self):
        """The context manager can set the flag to False."""
        with torch._library.triton.set_wrap_triton_enabled(False):
            self.assertFalse(torch._library.triton.is_wrap_triton_enabled())

    def test_set_wrap_triton_enabled_restores_on_exit(self):
        """The previous state is restored when the context manager exits."""
        original = torch._library.triton.is_wrap_triton_enabled()
        with torch._library.triton.set_wrap_triton_enabled(False):
            pass
        self.assertEqual(torch._library.triton.is_wrap_triton_enabled(), original)

    def test_set_wrap_triton_enabled_restores_on_exception(self):
        """The previous state is restored even if the context body raises."""
        original = torch._library.triton.is_wrap_triton_enabled()
        with self.assertRaises(RuntimeError):
            with torch._library.triton.set_wrap_triton_enabled(False):
                self.assertFalse(torch._library.triton.is_wrap_triton_enabled())
                raise RuntimeError("intentional")
        self.assertEqual(torch._library.triton.is_wrap_triton_enabled(), original)

    def test_set_wrap_triton_enabled_nested(self):
        """Nested context managers restore the correct previous state at each level."""
        with torch._library.triton.set_wrap_triton_enabled(False):
            self.assertFalse(torch._library.triton.is_wrap_triton_enabled())
            with torch._library.triton.set_wrap_triton_enabled(True):
                self.assertTrue(torch._library.triton.is_wrap_triton_enabled())
            self.assertFalse(torch._library.triton.is_wrap_triton_enabled())
        self.assertTrue(torch._library.triton.is_wrap_triton_enabled())

    def test_set_wrap_triton_enabled_affects_wrap_triton(self):
        """When disabled, wrap_triton returns the raw kernel; when enabled, a wrapper."""
        if not has_triton_package():
            self.skipTest("requires triton")

        import triton
        import triton.language as tl

        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        # Default: wrapping enabled -> wrap_triton returns a wrapper.
        wrapped = torch.library.wrap_triton(add_kernel)
        self.assertIsNot(wrapped, add_kernel)

        with torch._library.triton.set_wrap_triton_enabled(False):
            unwrapped = torch.library.wrap_triton(add_kernel)
            self.assertIs(unwrapped, add_kernel)


if __name__ == "__main__":
    run_tests()
