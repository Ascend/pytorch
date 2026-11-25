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
# Owner(s): ["oncall: profiler"]

import os
import tempfile
from subprocess import CompletedProcess
from unittest.mock import patch

from torch_npu.profiler.analysis.prof_view.cann_parse._cann_command import (
    build_command_log_path,
    save_command_output,
)
from torch_npu.testing.testcase import run_tests, TestCase


class TestCannCommand(TestCase):
    def test_build_command_log_path(self):
        log_path = build_command_log_path("/fake/path", "export")
        self.assertTrue(log_path.startswith(os.path.join("/fake/path", "logs")))
        file_name = os.path.basename(log_path)
        self.assertTrue(file_name.startswith("msprof_export_"))
        self.assertTrue(file_name.endswith(".log"))
        self.assertIn(str(os.getpid()), file_name)

    def test_build_command_log_path_differs_by_mode(self):
        export_path = build_command_log_path("/fake/path", "export")
        analyze_path = build_command_log_path("/fake/path", "analyze")
        self.assertNotEqual(export_path, analyze_path)
        self.assertIn("msprof_analyze_", os.path.basename(analyze_path))

    def test_save_command_output(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = os.path.join(tmp_dir, "logs", "msprof_export_1234.log")
            completed = CompletedProcess(
                args=["msprof"], returncode=0, stdout="export start\nexport done\n", stderr=""
            )
            save_command_output(log_path, ["msprof", "--export=on"], completed)
            with open(log_path, encoding="utf-8") as file:
                content = file.read()
            self.assertIn("[command] msprof --export=on", content)
            self.assertIn("[returncode] 0", content)
            self.assertIn("export done", content)

    def test_save_command_output_appends(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = os.path.join(tmp_dir, "logs", "msprof_export_1234.log")
            first = CompletedProcess(args=[], returncode=0, stdout="first\n", stderr="")
            second = CompletedProcess(args=[], returncode=1, stdout="second\n", stderr="error\n")
            save_command_output(log_path, ["msprof", "--export=on"], first)
            save_command_output(log_path, ["msprof", "--export=on", "--type=db"], second)
            with open(log_path, encoding="utf-8") as file:
                content = file.read()
            self.assertIn("first", content)
            self.assertIn("second", content)
            self.assertIn("[returncode] 1", content)
            self.assertIn("error", content)

    def test_save_command_output_bytes_and_none(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = os.path.join(tmp_dir, "logs", "msprof_analyze_1234.log")
            completed = CompletedProcess(args=[], returncode=0, stdout=b"bytes output", stderr=None)
            save_command_output(log_path, ["msprof"], completed)
            with open(log_path, encoding="utf-8") as file:
                content = file.read()
            self.assertIn("bytes output", content)

    def test_save_command_output_write_failure_does_not_raise(self):
        completed = CompletedProcess(args=[], returncode=0, stdout="output", stderr="")
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("builtins.open", side_effect=OSError("no space left")), patch(
                "torch_npu.profiler.analysis.prof_view.cann_parse._cann_command.print_warn_msg"
            ) as mock_warn:
                save_command_output(
                    os.path.join(tmp_dir, "logs", "msprof_export_1234.log"),
                    ["msprof"],
                    completed,
                )
                mock_warn.assert_called_once()


if __name__ == "__main__":
    run_tests()
