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
Add validation cases for torch.set_warn_always on NPU:
1. PyTorch community lacks complete and direct validation for this API.
2. This file validates state switching, warning behavior, and invalid inputs (extendable).
"""

import subprocess
import sys
import textwrap
import warnings

import numpy as np
import torch
from torch.testing._internal.common_utils import TestCase, run_tests


class TestSetWarnAlways(TestCase):

    def setUp(self):
        super().setUp()
        self.original_state = torch.is_warn_always_enabled()
        self.addCleanup(torch.set_warn_always, self.original_state)

    def test_state_switching_and_return_value(self):
        result = torch.set_warn_always(True)
        self.assertIsNone(result)
        self.assertTrue(torch.is_warn_always_enabled())

        result = torch.set_warn_always(False)
        self.assertIsNone(result)
        self.assertFalse(torch.is_warn_always_enabled())

    def test_warn_always_emits_repeated_warnings(self):
        array = np.arange(10)
        array.flags.writeable = False
        message = "not writable"
        torch.set_warn_always(True)

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            # NumPy conversion is CPU-only and provides a stable TORCH_WARN_ONCE source.
            torch.from_numpy(array)
            torch.from_numpy(array)

        matched = [record for record in records if message in str(record.message)]
        self.assertEqual(len(matched), 2)
        torch.set_warn_always(False)

    def test_warn_once_behavior_when_disabled(self):
        code = textwrap.dedent(
            """
            import warnings

            import numpy as np
            import torch

            array = np.arange(10)
            array.flags.writeable = False
            torch.set_warn_always(False)
            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter("always")
                # NumPy conversion is CPU-only and provides a stable TORCH_WARN_ONCE source.
                torch.from_numpy(array)
                torch.from_numpy(array)

            message = "not writable"
            matched = [record for record in records if message in str(record.message)]
            raise SystemExit(0 if len(matched) == 1 else 1)
            """
        )
        process = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(process.returncode, 0, process.stderr)

    def test_invalid_inputs_preserve_state(self):
        torch.set_warn_always(True)

        for value in (1, 0, None, "true", object()):
            with self.subTest(value=value):
                with self.assertRaises((RuntimeError, TypeError)):
                    torch.set_warn_always(value)
                self.assertTrue(torch.is_warn_always_enabled())

        with self.assertRaises(TypeError):
            torch.set_warn_always(b=False)
        with self.assertRaises(TypeError):
            torch.set_warn_always()
        with self.assertRaises(TypeError):
            torch.set_warn_always(True, False)
        self.assertTrue(torch.is_warn_always_enabled())


if __name__ == "__main__":
    run_tests()
