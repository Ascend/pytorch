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
Add validation cases for torch.jit._state._clear_class_state API:
1. PyTorch community lacks sufficient direct tests for the internal JIT
   class state clearing function.
2. This file validates the behavior of _clear_class_state, ensuring that
   script class registrations are correctly cleared and that subsequent
   script compilations are not affected by stale state, which is important
   for test isolation and stability.
"""

import torch
from torch.jit import _state
from torch.testing._internal.common_utils import run_tests, TestCase


class TestClearClassState(TestCase):
    """Test case for torch.jit._state._clear_class_state."""

    def _verify_class_clear(self):
        """Helper to encapsulate the core verification logic for SCA compliance."""
        # Backup internal states
        saved_script_classes = dict(_state._script_classes)
        saved_name_to_pyclass = dict(_state._name_to_pyclass)

        try:
            @torch.jit.script
            class MyClass:
                def __init__(self, val: int):
                    self.val = val

            old_cls = MyClass
            self.assertIsNotNone(
                _state._get_script_class(MyClass),
                'Class should be present in JIT state after scripting.'
            )

            _state._clear_class_state()

            self.assertIsNone(
                _state._get_script_class(old_cls),
                'Class should be removed from JIT state after clearing.'
            )

            @torch.jit.script
            class MyClass:
                def __init__(self, val: int, extra: int = 0):
                    self.val = val
                    self.extra = extra

                def get_extra(self) -> int:
                    return self.extra

            self.assertIsNotNone(
                _state._get_script_class(MyClass),
                'New class with same name should be registered after re-scripting.'
            )

            obj = MyClass(10, 20)
            self.assertEqual(obj.val, 10)
            self.assertEqual(obj.extra, 20)
            self.assertEqual(obj.get_extra(), 20)

            self.assertIsNone(
                _state._get_script_class(old_cls),
                'Old class object should remain absent after re-scripting.'
            )

        finally:
            # Restore internal states
            _state._script_classes.clear()
            _state._script_classes.update(saved_script_classes)
            _state._name_to_pyclass.clear()
            _state._name_to_pyclass.update(saved_name_to_pyclass)

    def test_clear_class_state(self):
        """Main test that calls the helper."""
        self._verify_class_clear()


if __name__ == '__main__':
    run_tests()
