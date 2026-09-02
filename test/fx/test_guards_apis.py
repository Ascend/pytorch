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
# Owner(s): ["module: fx"]
"""
Add validation cases for torch._guards.active_fake_mode API.

API Introduction:
torch._guards.active_fake_mode() returns the currently active FakeTensorMode
instance from the torch dispatch mode stack, or None if no FakeTensorMode is
active. It is used internally by PyTorch's compiler/tracing infrastructure to
query whether tensor operations are currently executed under fake tensor tracing.

Test Design:
1. PyTorch community lacks sufficient and direct API validations for this API,
   so this file is added.
2. This file validates torch._guards.active_fake_mode in the following scenarios:
   - returns None when no FakeTensorMode is active
   - returns the correct FakeTensorMode instance when one is active
   - returns None after the FakeTensorMode context exits
   - correctly tracks nested FakeTensorMode contexts (inner/outer identity)
   - correctly identifies fake tensors created within the context
"""
import torch
from torch._guards import active_fake_mode
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.common_utils import TestCase, run_tests
acc = torch.accelerator.current_accelerator()
if acc is None:
    raise RuntimeError("No available accelerator. This test requires an NPU/GPU.")
device_type = acc.type


class TestActiveFakeMode(TestCase):
    """Tests for torch._guards.active_fake_mode API."""

    def test_active_fake_mode_returns_none_outside_context(self):
        result = active_fake_mode()
        self.assertIsNone(result)

    def test_active_fake_mode_returns_mode_inside_context(self):
        with FakeTensorMode() as mode:
            result = active_fake_mode()
            self.assertIsNotNone(result)
            self.assertIsInstance(result, FakeTensorMode)

    def test_active_fake_mode_returns_none_after_context_exits(self):
        with FakeTensorMode():
            pass
        result = active_fake_mode()
        self.assertIsNone(result)

    def test_active_fake_mode_nested_context(self):
        with FakeTensorMode() as outer:
            result_outer = active_fake_mode()
            self.assertIs(result_outer, outer)
            with FakeTensorMode() as inner:
                result_inner = active_fake_mode()
                self.assertIs(result_inner, inner)
            result_after_inner = active_fake_mode()
            self.assertIs(result_after_inner, outer)
        result_after_all = active_fake_mode()
        self.assertIsNone(result_after_all)

    def test_active_fake_mode_with_fake_tensor(self):
        with FakeTensorMode() as mode:
            t = torch.empty(2, 3).to(device_type)
            result = active_fake_mode()
            self.assertIs(result, mode)
            self.assertTrue(isinstance(t, torch._subclasses.fake_tensor.FakeTensor))


if __name__ == '__main__':
    run_tests()
