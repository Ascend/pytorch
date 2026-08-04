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
Add validation cases for torch._logging.set_logs API:
1. PyTorch community lacks sufficient and direct API validations for
   torch._logging.set_logs, so this file is added.
2. This file validates the log level configuration, artifact enabling,
   modules parameter handling, invalid input validation, and environment
   variable precedence for torch._logging.set_logs (extendable).
"""

import logging
import os
import unittest.mock

import torch
import torch._logging._internal
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.logging_utils import preserve_log_state


class TestLibraryLogging(TestCase):
    """Test torch._logging.set_logs."""

    def test_set_logs_default_clears_state(self):
        """Calling set_logs() with no arguments resets the logging state."""
        with preserve_log_state():
            torch._logging.set_logs(dynamo=logging.DEBUG, graph_code=True)
            self.assertTrue(
                torch._logging._internal.log_state.is_artifact_enabled("graph_code")
            )
            torch._logging.set_logs()
            self.assertFalse(
                torch._logging._internal.log_state.is_artifact_enabled("graph_code")
            )
            self.assertEqual(
                list(torch._logging._internal.log_state.get_log_level_pairs()), []
            )

    def test_set_logs_enable_component(self):
        """set_logs can set the log level for registered components."""
        with preserve_log_state():
            torch._logging.set_logs(dynamo=logging.DEBUG)
            pairs = dict(torch._logging._internal.log_state.get_log_level_pairs())
            self.assertIn("torch._dynamo", pairs)
            self.assertEqual(pairs["torch._dynamo"], logging.DEBUG)
            self.assertEqual(logging.getLogger("torch._dynamo").level, logging.DEBUG)

    def test_set_logs_enable_artifact(self):
        """set_logs can enable registered artifacts."""
        with preserve_log_state():
            torch._logging.set_logs(graph_code=True)
            self.assertTrue(
                torch._logging._internal.log_state.is_artifact_enabled("graph_code")
            )

    def test_set_logs_modules(self):
        """set_logs supports registered aliases through the modules argument."""
        with preserve_log_state():
            torch._logging.set_logs(modules={"dynamo": logging.INFO})
            pairs = dict(torch._logging._internal.log_state.get_log_level_pairs())
            self.assertIn("torch._dynamo", pairs)
            self.assertEqual(pairs["torch._dynamo"], logging.INFO)

    def test_set_logs_invalid_artifact_value(self):
        """Passing a non-bool value for an artifact raises ValueError."""
        with preserve_log_state():
            with self.assertRaises(ValueError):
                torch._logging.set_logs(graph_code=5)

    def test_set_logs_invalid_log_level(self):
        """Passing an unrecognized log level raises ValueError."""
        with preserve_log_state():
            with self.assertRaises(ValueError):
                torch._logging.set_logs(dynamo=999)

    def test_set_logs_invalid_module_name(self):
        """Passing an unrecognized module name via modules raises ValueError."""
        with preserve_log_state():
            with self.assertRaises(ValueError):
                torch._logging.set_logs(modules={"not_a_real_thing": logging.INFO})

    def test_set_logs_env_var_precedence(self):
        """When TORCH_LOGS is set, set_logs does nothing."""
        with unittest.mock.patch.dict(os.environ, {"TORCH_LOGS": "dynamo"}):
            with preserve_log_state():
                torch._logging.set_logs(dynamo=logging.DEBUG)
                pairs = dict(torch._logging._internal.log_state.get_log_level_pairs())
                self.assertNotIn("torch._dynamo", pairs)


if __name__ == "__main__":
    run_tests()
