# Copyright (c) 2026 Huawei Technologies Co., Ltd
# Licensed under the BSD 3-Clause License.

import os
import subprocess
import sys
import unittest

import torch

from torch_npu.testing.testcase import TestCase, run_tests


LOG_LEVEL_ENV = "ASCEND_GLOBAL_LOG_LEVEL"
SUBPROCESS_TIMEOUT = 30

# OptionsManager::isACLGlobalLogOn caches the parsed level in a function-local
# static on the first ASCEND_LOG* call. Every case must therefore run in a
# fresh process, with the environment set before torch_npu is imported.
#
# Before the fix, an invalid value made TORCH_CHECK throw inside the static
# initializer, whose error path (formatErrorCode -> GetDevice ->
# is_lazy_set_device -> ASCEND_LOGW) re-enters isACLGlobalLogOn: same-thread
# magic-static re-entry hangs the process. With a plain-text TORCH_CHECK the
# failure mode is instead std::terminate (core dump) during unwinding. Either
# way the child never exits cleanly, so "returncode == 0" is the regression
# assertion for all invalid values.
# The historical hang triggers during import itself (the first ASCEND_LOG*
# call runs the static initializer), so a bare import is sufficient to
# reproduce/verify; no device work is needed.
_CHILD_SCRIPT = r"""
import torch
import torch_npu
print("CASE_SUCCESS")
"""


@unittest.skipIf(not torch.npu.is_available(), "requires NPU")
class TestAscendGlobalLogLevel(TestCase):
    # End-to-end coverage for the ASCEND_GLOBAL_LOG_LEVEL fallback
    # behaviour, one fresh subprocess per value (the parsed level is cached
    # in a function-local static). See test methods for the covered reject
    # branches.

    @staticmethod
    def _run_child(env_value):
        env = os.environ.copy()
        # Slog output must not pollute the child stdout used for assertions.
        env.pop("ASCEND_SLOG_PRINT_TO_STDOUT", None)
        if env_value is None:
            env.pop(LOG_LEVEL_ENV, None)
        else:
            env[LOG_LEVEL_ENV] = env_value
        return subprocess.run(
            [sys.executable, "-c", _CHILD_SCRIPT],
            env=env,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT,
            check=False,
        )

    def _assert_child_ok(self, env_value):
        result = self._run_child(env_value)
        output = (result.stdout or "") + (result.stderr or "")
        self.assertEqual(
            result.returncode,
            0,
            f"env={LOG_LEVEL_ENV}={env_value!r} must not hang or abort\n{output}",
        )
        self.assertIn("CASE_SUCCESS", output)

    def test_unset_falls_back_to_default(self):
        self._assert_child_ok(None)

    def test_valid_levels_are_accepted(self):
        # 0-4 are the documented levels (debug/info/warning/error/null);
        # valid values must keep working exactly as before the fix.
        for value in ("0", "1", "2", "3", "4"):
            with self.subTest(value=value):
                self._assert_child_ok(value)

    def test_invalid_values_fall_back_to_default(self):
        # One representative per reject branch of the parser:
        #   "5"              - out of range (original hang trigger, issue 5342)
        #   "-1"             - negative
        #   "99999999999999" - beyond int32, caught by the round-trip check
        #   "9" * 30         - beyond int64, strtol saturates to LONG_MAX and
        #                      the round-trip check rejects it
        #   "abc"            - no conversion (strtol endptr == nptr)
        #   "3x"             - trailing garbage (*endptr != '\0')
        #   ""               - empty must not enable debug via strtol("") == 0
        for value in ("5", "-1", "99999999999999", "9" * 30, "abc", "3x", ""):
            with self.subTest(value=value):
                self._assert_child_ok(value)

    def test_int32_narrowing_falls_back_to_default_level(self):
        # A plain "process exits cleanly" assertion cannot distinguish "fell
        # back to error" from "narrowing bypassed validation and enabled
        # debug" -- both exit 0. Observe the effect instead: with slog printed
        # to stdout, 2**32 (wraps to 0 = debug when narrowed to int32) must
        # produce the same log volume as the default error(3) level, not the
        # debug(0) level it would wrap to.
        default_lines = self._count_slog_lines("3")
        debug_lines = self._count_slog_lines("0")
        # Guard the measurement itself: if debug(0) does not emit more lines
        # than error(3), the count is not level-sensitive and the comparison
        # below would pass vacuously.
        self.assertGreater(
            debug_lines,
            default_lines,
            "slog line count is not level-sensitive: debug(0) must emit "
            f"more lines than error(3) ({debug_lines} vs {default_lines})",
        )
        wrapped_lines = self._count_slog_lines("4294967296")
        self.assertEqual(
            default_lines,
            wrapped_lines,
            "2**32 wraps to 0 (debug) when narrowed to int32 and is not "
            f"rejected: {default_lines} (default) vs {wrapped_lines} "
            f"(wrapped) log lines, debug baseline is {debug_lines}",
        )

    @staticmethod
    def _count_slog_lines(env_value):
        # Child with slog-to-stdout enabled. Unlike _CHILD_SCRIPT this one
        # keeps set_device: import + set_device emit LOGI/LOGD lines, so the
        # count is nonzero and level-sensitive.
        script = (
            "import os\n"
            'os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "1"\n'
            "import torch\n"
            "import torch_npu\n"
            "torch.npu.set_device(0)\n"
        )
        env = os.environ.copy()
        env.pop("ASCEND_SLOG_PRINT_TO_STDOUT", None)
        env[LOG_LEVEL_ENV] = env_value
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT,
            check=False,
        )
        # Slog lines start with "[INFO]" / "[WARNING]" / "[ERROR]" markers.
        return sum(
            1
            for line in (result.stdout or "").splitlines()
            if "[INFO]" in line or "[WARNING]" in line or "[ERROR]" in line
        )


if __name__ == "__main__":
    run_tests()
