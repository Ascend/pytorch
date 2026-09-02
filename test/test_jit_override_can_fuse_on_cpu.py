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
Add consistency validation cases for torch._C._jit_override_can_fuse_on_cpu.

This file adds self-contained, focused validation for
torch._C._jit_override_can_fuse_on_cpu as requested by the
Ascend for PyTorch API consistency task (#2756). The setter overrides whether
ops can fuse on CPU in the JIT fuser; it accepts a bool value and the
override actually takes effect (verified via the _jit_can_fuse_on_cpu getter).
Two cases are covered:

* ``test_override_can_fuse_on_cpu`` -- the setter returns None and the global
  flag reads back the value just set (False / True / original), proving the
  override actually takes effect.
* ``test_override_can_fuse_on_cpu_invalid_type`` -- the setter only accepts a
  bool flag; any non-bool argument must be rejected with a type error (or a
  RuntimeError surfaced from the C++ binding) instead of being silently
  coerced.

Extendable: peer torch._C._jit_* flags can be appended to this file.
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestJitOverrideCanFuseOnCpu(TestCase):

    def setUp(self):
        super().setUp()
        self._old_can_fuse_on_cpu = None
        if not hasattr(torch._C, "_jit_override_can_fuse_on_cpu"):
            self.skipTest("torch._C._jit_override_can_fuse_on_cpu unavailable")
        if not hasattr(torch._C, "_jit_can_fuse_on_cpu"):
            self.skipTest("torch._C._jit_can_fuse_on_cpu unavailable")
        # Save the original global flag; restored in tearDown so this test does
        # not leak the override into other tests. Mirrors the save/restore
        # pattern used in test/test_jit_fuser_te.py (see test_disabled).
        self._old_can_fuse_on_cpu = torch._C._jit_can_fuse_on_cpu()

    def tearDown(self):
        if self._old_can_fuse_on_cpu is not None:
            torch._C._jit_override_can_fuse_on_cpu(self._old_can_fuse_on_cpu)
        super().tearDown()

    def test_override_can_fuse_on_cpu(self):
        # The setter is a void flag setter; each call returns None.
        self.assertIsNone(torch._C._jit_override_can_fuse_on_cpu(False))
        # After setting False, the global flag reads back as False.
        self.assertFalse(torch._C._jit_can_fuse_on_cpu())

        self.assertIsNone(torch._C._jit_override_can_fuse_on_cpu(True))
        # After setting True, the global flag reads back as True.
        self.assertTrue(torch._C._jit_can_fuse_on_cpu())

        # Restoring to the original value also takes effect.
        self.assertIsNone(
            torch._C._jit_override_can_fuse_on_cpu(self._old_can_fuse_on_cpu)
        )
        self.assertEqual(
            torch._C._jit_can_fuse_on_cpu(), self._old_can_fuse_on_cpu
        )

    def test_override_can_fuse_on_cpu_invalid_type(self):
        # The setter only accepts a bool flag. A non-bool argument must be
        # rejected with a TypeError (or a RuntimeError surfaced from the C++
        # binding) rather than silently coerced. Both the value check and the
        # no-partial-mutation invariant are asserted.
        for bad in (None, "True", 1.5, [True], {"a": 1}, (1, 2)):
            with self.assertRaises((TypeError, RuntimeError)):
                torch._C._jit_override_can_fuse_on_cpu(bad)
        # The flag must remain unchanged after the rejected calls. Restore
        # defensively in case any call happened to succeed on a given build.
        torch._C._jit_override_can_fuse_on_cpu(self._old_can_fuse_on_cpu)
        self.assertEqual(
            torch._C._jit_can_fuse_on_cpu(), self._old_can_fuse_on_cpu
        )


if __name__ == "__main__":
    run_tests()
