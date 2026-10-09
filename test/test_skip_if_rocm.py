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
Add validation cases for torch.testing._internal.common_utils.skipIfRocm on NPU:

1. PyTorch upstream only uses skipIfRocm indirectly (for example
   TestImports.test_circular_dependencies in test/test_testing.py), so the
   decorator has no direct coverage of its run path, skip path, skip reason
   and the two supported usage forms.
2. NPU is not a ROCm build, so a decorated case must run normally; once
   TEST_WITH_ROCM is enabled the same case must be reported as skipped with the
   expected reason. Both the bare form (@skipIfRocm) and the parameterized form
   (@skipIfRocm(msg=...)) are covered.
"""

import unittest
from unittest import mock

import torch
from torch.testing._internal.common_utils import TestCase, run_tests, skipIfRocm

COMMON_UTILS = "torch.testing._internal.common_utils"
DEFAULT_MSG = "test doesn't currently work on the ROCm stack"
DEFAULT_REASON = f"skipIfRocm: {DEFAULT_MSG}"
CUSTOM_MSG = "custom skip reason"

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestSkipIfRocm(TestCase):

    @skipIfRocm
    def test_decorated_test_method_runs_on_npu(self):
        x = torch.randn(2, 2, device=device_type)
        self.assertEqual(x + 1, torch.add(x, 1))

    def test_bare_decorator_runs_on_npu(self):
        @skipIfRocm
        def add_one(x):
            return x + 1

        x = torch.randn(2, 2, device=device_type)
        self.assertEqual(add_one(x), x + 1)

    def test_msg_decorator_runs_on_npu(self):
        @skipIfRocm(msg=CUSTOM_MSG)
        def add_one(x):
            return x + 1

        x = torch.randn(2, 2, device=device_type)
        self.assertEqual(add_one(x), x + 1)

    def test_non_rocm_build_keeps_case_enabled(self):
        self.assertIsNone(torch.version.hip, "NPU is not a ROCm build")

        @skipIfRocm
        def add_one(x):
            return x + 1

        self.assertFalse(getattr(add_one, "__unittest_skip__", False))

    def test_skip_when_rocm_enabled(self):
        self._assert_skipped_on_rocm(msg=None, expected_reason=DEFAULT_REASON)

    def test_skip_reason_uses_custom_msg(self):
        self._assert_skipped_on_rocm(msg=CUSTOM_MSG, expected_reason=f"skipIfRocm: {CUSTOM_MSG}")

    def _assert_skipped_on_rocm(self, msg, expected_reason):
        """A decorated case must be reported as skipped once TEST_WITH_ROCM is on."""
        with mock.patch(f"{COMMON_UTILS}.TEST_WITH_ROCM", True):
            if msg is None:

                @skipIfRocm
                def decorated(x):
                    return x + 1
            else:

                @skipIfRocm(msg=msg)
                def decorated(x):
                    return x + 1

            if getattr(decorated, "__unittest_skip__", False):
                # Class targets of the decorator are marked at decoration time.
                self.assertEqual(decorated.__unittest_skip_why__, expected_reason)
            else:
                # Function targets raise SkipTest when the case is executed.
                with self.assertRaises(unittest.SkipTest) as ctx:
                    decorated(torch.randn(1, device=device_type))
                self.assertEqual(str(ctx.exception), expected_reason)

            class RocmSkippedCase(unittest.TestCase):

                @skipIfRocm(msg=msg if msg is not None else DEFAULT_MSG)
                def test_case(self):
                    self.fail("a decorated case must not run when TEST_WITH_ROCM is enabled")

            result = unittest.TestResult()
            RocmSkippedCase("test_case").run(result)
            self.assertEqual(result.testsRun, 1)
            self.assertEqual(result.failures, [])
            self.assertEqual(result.errors, [])
            self.assertEqual(len(result.skipped), 1, result.skipped)
            self.assertIn(expected_reason, result.skipped[0][1])


if __name__ == "__main__":
    run_tests()
