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
from unittest.mock import MagicMock, patch

from torch_npu.profiler.analysis.prof_common_func._constant import Constant
from torch_npu.profiler.analysis.prof_view.cann_parse._cann_export import (
    CANNExportParser,
    CANNTimelineParser,
)
from torch_npu.testing.testcase import run_tests, TestCase


class TestCannExport(TestCase):
    def test_cann_export_parser_run_db_export_success(self):
        with (
            patch("os.path.isdir", return_value=True),
            patch("subprocess.run") as mock_run,
            patch(
                "torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path",
                return_value="/fake/cann/path",
            ),
            patch(
                "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.shutil.which",
                return_value="/usr/bin/msprof",
            ),
            patch(
                "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.ProfilerConfig"
            ),
            patch(
                "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.ProfilerLogger"
            ),
            patch(
                "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.CANNExportParser._check_msprof_environment"
            ),
            patch(
                "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.CANNExportParser._check_prof_data_size"
            ),
            patch(
                "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.save_command_output"
            ),
        ):
            mock_run.return_value.returncode = 0
            parser = CANNExportParser(
                "test", {"profiler_path": "/fake/path", Constant.EXPORT_TYPE: ["db"]}
            )
            parser.msprof_path = "/usr/bin/msprof"
            result = parser.run({})
            self.assertEqual(result, (Constant.SUCCESS, None))

    def test_cann_export_parser_text_export_uses_type_all(self):
        result, mock_run = self._run_text_export(["text"], True)
        self.assertEqual(result, (Constant.SUCCESS, None))
        self.assertIn("--type=all", mock_run.call_args_list[0][0][0])

    def test_cann_export_parser_text_export_without_type_all(self):
        result, mock_run = self._run_text_export(["text"], False)
        self.assertEqual(result, (Constant.SUCCESS, None))
        self.assertNotIn("--type=all", mock_run.call_args_list[0][0][0])

    def test_timeline_parser_has_msprof_timeline(self):
        self.assertTrue(CANNTimelineParser._has_msprof_timeline(["msprof_1234.json"]))
        self.assertTrue(
            CANNTimelineParser._has_msprof_timeline(
                ["op_summary_1234.csv", "msprof_slice_1234.json"]
            )
        )
        self.assertFalse(CANNTimelineParser._has_msprof_timeline(["op_summary_1234.csv"]))
        self.assertFalse(CANNTimelineParser._has_msprof_timeline(["1234_msprof.json"]))
        self.assertFalse(CANNTimelineParser._has_msprof_timeline([]))

    def test_timeline_parser_find_finish_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = os.path.join(tmp_dir, "mindstudio_profiler_output")
            os.makedirs(output_dir)
            output_done = os.path.join(output_dir, "finish.txt")
            with open(output_done, "w") as file:
                file.write("msprof_1234.json\n")
            with patch(
                "torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path",
                return_value=tmp_dir,
            ):
                parser = CANNTimelineParser(
                    "test", {"profiler_path": "/fake/path", Constant.EXPORT_TYPE: ["text"]}
                )
                self.assertEqual(parser._find_finish_file(), output_done)

    def test_timeline_parser_run_with_export_all_done(self):
        result = self._run_timeline_parser("op_summary_1234.csv\nmsprof_1234.json\n")
        self.assertEqual(result, (Constant.SUCCESS, None))

    def test_timeline_parser_run_with_export_all_finished_mark(self):
        result = self._run_timeline_parser("op_summary_1234.csv\ndone\n")
        self.assertEqual(result, (Constant.SUCCESS, None))

    def test_timeline_parser_run_with_timeline_and_finished_mark(self):
        result = self._run_timeline_parser(
            "op_summary_1234.csv\nmsprof_1234.json\ndone\n"
        )
        self.assertEqual(result, (Constant.SUCCESS, None))

    def _run_text_export(self, export_type: list, support_export_all: bool, returncode: int = 0):
        with (
            patch("os.path.isdir", return_value=True),
            patch("subprocess.run") as mock_run,
            patch(
                "torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path",
                return_value="/fake/cann/path",
            ),
            patch(
                "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.shutil.which",
                return_value="/usr/bin/msprof",
            ),
            patch(
                "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.ProfilerConfig"
            ),
            patch(
                "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.ProfilerLogger"
            ),
            patch(
                "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.CANNExportParser._check_msprof_environment"
            ),
            patch(
                "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.CANNExportParser._check_prof_data_size"
            ),
            patch(
                "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.CannPackageManager.is_support_export_all",
                return_value=support_export_all,
            ),
            patch(
                "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.save_command_output"
            ),
        ):
            mock_run.return_value.returncode = returncode
            parser = CANNExportParser(
                "test", {"profiler_path": "/fake/path", Constant.EXPORT_TYPE: export_type}
            )
            parser.msprof_path = "/usr/bin/msprof"
            return parser.run({}), mock_run

    def test_cann_export_parser_text_export_failed(self):
        result, _ = self._run_text_export(["text"], True, returncode=1)
        self.assertEqual(result, (Constant.FAIL, None))

    def test_cann_export_parser_db_export_failed(self):
        result, _ = self._run_text_export(["db"], False, returncode=1)
        self.assertEqual(result, (Constant.FAIL, None))

    def _run_timeline_parser(self, output_done_content: str):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = os.path.join(tmp_dir, "mindstudio_profiler_output")
            os.makedirs(output_dir)
            with open(os.path.join(output_dir, "finish.txt"), "w") as file:
                file.write(output_done_content)
            with (
                patch(
                    "torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path",
                    return_value=tmp_dir,
                ),
                patch(
                    "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.CannPackageManager.is_support_export_all",
                    return_value=True,
                ),
                patch(
                    "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.ProfilerConfig"
                ),
                patch(
                    "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.ProfilerLogger"
                ),
            ):
                parser = CANNTimelineParser(
                    "test", {"profiler_path": "/fake/path", Constant.EXPORT_TYPE: ["text"]}
                )
                return parser.run({})

    def test_timeline_parser_run_without_cann_path(self):
        with (
            patch(
                "torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path",
                return_value="/nonexistent/path",
            ),
            patch("os.path.isdir", return_value=False),
            patch(
                "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.ProfilerLogger"
            ),
        ):
            parser = CANNTimelineParser(
                "test", {"profiler_path": "/fake/path", Constant.EXPORT_TYPE: ["text"]}
            )
            self.assertEqual(parser.run({}), (Constant.SUCCESS, None))

    def test_timeline_parser_legacy_waits_for_summary_csv(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = os.path.join(tmp_dir, "mindstudio_profiler_output")
            os.makedirs(output_dir)
            with open(os.path.join(output_dir, "op_summary_1234.csv"), "w"):
                pass
            with (
                patch(
                    "torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path",
                    return_value=tmp_dir,
                ),
                patch(
                    "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.CannPackageManager.is_support_export_all",
                    return_value=False,
                ),
                patch(
                    "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.ProfilerConfig"
                ),
                patch(
                    "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.ProfilerLogger"
                ),
            ):
                parser = CANNTimelineParser(
                    "test", {"profiler_path": "/fake/path", Constant.EXPORT_TYPE: ["text"]}
                )
                self.assertEqual(parser.run({}), (Constant.SUCCESS, None))

    def test_timeline_parser_waits_for_db_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, "msprof_1234.db"), "w"):
                pass
            with (
                patch(
                    "torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path",
                    return_value=tmp_dir,
                ),
                patch(
                    "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.ProfilerConfig"
                ),
                patch(
                    "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.ProfilerLogger"
                ),
            ):
                parser = CANNTimelineParser(
                    "test", {"profiler_path": "/fake/path", Constant.EXPORT_TYPE: ["db"]}
                )
                self.assertEqual(parser.run({}), (Constant.SUCCESS, None))

    def test_timeline_parser_db_ignores_finish_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = os.path.join(tmp_dir, "mindstudio_profiler_output")
            os.makedirs(output_dir)
            # db 路径保持老逻辑：finish.txt 不参与判定，没有 db 文件就继续等
            with open(os.path.join(output_dir, "finish.txt"), "w") as file:
                file.write("msprof_1234.db\ndone\n")
            with (
                patch(
                    "torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path",
                    return_value=tmp_dir,
                ),
                patch(
                    "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.ProfilerConfig"
                ),
                patch(
                    "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.ProfilerLogger"
                ),
                patch(
                    "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.time.sleep",
                    side_effect=InterruptedError,
                ),
            ):
                parser = CANNTimelineParser(
                    "test", {"profiler_path": "/fake/path", Constant.EXPORT_TYPE: ["db"]}
                )
                self.assertEqual(parser.run({}), (Constant.FAIL, None))

    def test_timeline_parser_keeps_waiting_without_json_and_finish_mark(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = os.path.join(tmp_dir, "mindstudio_profiler_output")
            os.makedirs(output_dir)
            with open(os.path.join(output_dir, "finish.txt"), "w") as file:
                file.write("op_summary_1234.csv\n")
            with (
                patch(
                    "torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path",
                    return_value=tmp_dir,
                ),
                patch(
                    "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.CannPackageManager.is_support_export_all",
                    return_value=True,
                ),
                patch(
                    "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.ProfilerConfig"
                ),
                patch(
                    "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.ProfilerLogger"
                ),
                patch(
                    "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.time.sleep",
                    side_effect=InterruptedError,
                ),
            ):
                parser = CANNTimelineParser(
                    "test", {"profiler_path": "/fake/path", Constant.EXPORT_TYPE: ["text"]}
                )
                # 只有 csv、既没有 msprof json 也没有 done：应当继续等待而不是提前放行
                self.assertEqual(parser.run({}), (Constant.FAIL, None))

    def test_timeline_parser_get_finish_lines_read_failure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = os.path.join(tmp_dir, "mindstudio_profiler_output")
            os.makedirs(output_dir)
            with open(os.path.join(output_dir, "finish.txt"), "w"):
                pass
            with (
                patch(
                    "torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path",
                    return_value=tmp_dir,
                ),
                patch(
                    "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.FileManager.file_read_all",
                    side_effect=OSError("read failed"),
                ),
                patch(
                    "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.ProfilerLogger"
                ),
            ):
                parser = CANNTimelineParser(
                    "test", {"profiler_path": "/fake/path", Constant.EXPORT_TYPE: ["text"]}
                )
                parser.logger = MagicMock()
                self.assertIsNone(parser._get_finish_lines())
                parser.logger.warning.assert_called_once()

    def test_timeline_parser_find_finish_file_rate_limited(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = os.path.join(tmp_dir, "mindstudio_profiler_output")
            os.makedirs(output_dir)
            output_done = os.path.join(output_dir, "finish.txt")
            with open(output_done, "w"):
                pass
            with patch(
                "torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path",
                return_value=tmp_dir,
            ):
                parser = CANNTimelineParser(
                    "test", {"profiler_path": "/fake/path", Constant.EXPORT_TYPE: ["text"]}
                )
                self.assertEqual(parser._find_finish_file(), output_done)
                # 缓存被清空、但距上次查找不足 1s：本轮不再遍历目录
                parser._output_done_path = ""
                self.assertEqual(parser._find_finish_file(), "")

    def test_cann_export_parser_saves_command_output(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                patch("os.path.isdir", return_value=True),
                patch("subprocess.run") as mock_run,
                patch(
                    "torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path",
                    return_value="/fake/cann/path",
                ),
                patch(
                    "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.shutil.which",
                    return_value="/usr/bin/msprof",
                ),
                patch(
                    "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.ProfilerConfig"
                ),
                patch(
                    "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.ProfilerLogger"
                ),
                patch(
                    "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.CANNExportParser._check_msprof_environment"
                ),
                patch(
                    "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.CANNExportParser._check_prof_data_size"
                ),
                patch(
                    "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.CannPackageManager.is_support_export_all",
                    return_value=True,
                ),
                patch(
                    "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.save_command_output"
                ) as mock_save,
            ):
                mock_run.return_value.returncode = 0
                parser = CANNExportParser(
                    "test", {"profiler_path": tmp_dir, Constant.EXPORT_TYPE: ["text"]}
                )
                parser.msprof_path = "/usr/bin/msprof"
                parser.run({})
        mock_save.assert_called_once()
        log_path, cmd_list = mock_save.call_args[0][0], mock_save.call_args[0][1]
        self.assertIn("msprof_export_", os.path.basename(log_path))
        self.assertIn("--type=all", cmd_list)

    def test_timeline_parser_get_finish_lines_without_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir, patch(
            "torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path",
            return_value=tmp_dir,
        ), patch(
            "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.ProfilerLogger"
        ):
            parser = CANNTimelineParser(
                "test", {"profiler_path": "/fake/path", Constant.EXPORT_TYPE: ["text"]}
            )
            self.assertIsNone(parser._get_finish_lines())

    def test_timeline_parser_find_finish_file_cache_hit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = os.path.join(tmp_dir, "mindstudio_profiler_output")
            os.makedirs(output_dir)
            output_done = os.path.join(output_dir, "finish.txt")
            with open(output_done, "w"):
                pass
            with patch(
                "torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path",
                return_value=tmp_dir,
            ):
                parser = CANNTimelineParser(
                    "test", {"profiler_path": "/fake/path", Constant.EXPORT_TYPE: ["text"]}
                )
                self.assertEqual(parser._find_finish_file(), output_done)
                # 第二次直接命中缓存，仍返回同一路径
                self.assertEqual(parser._find_finish_file(), output_done)

    def test_timeline_parser_find_finish_file_without_cann_path(self):
        with patch(
            "torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path",
            return_value="",
        ), patch(
            "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.ProfilerLogger"
        ):
            parser = CANNTimelineParser(
                "test", {"profiler_path": "/fake/path", Constant.EXPORT_TYPE: ["text"]}
            )
            self.assertEqual(parser._find_finish_file(), "")

    def test_timeline_parser_does_not_search_beyond_next_level(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            deep_dir = os.path.join(tmp_dir, "host", "data")
            os.makedirs(deep_dir)
            with open(os.path.join(deep_dir, "finish.txt"), "w"):
                pass
            with patch(
                "torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path",
                return_value=tmp_dir,
            ), patch(
                "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.ProfilerLogger"
            ):
                parser = CANNTimelineParser(
                    "test", {"profiler_path": "/fake/path", Constant.EXPORT_TYPE: ["text"]}
                )
                # 只查 PROF 本身和下一级目录，更深层的同名文件不参与
                self.assertEqual(parser._find_finish_file(), "")

    def test_cann_export_parser_init(self):
        with (
            patch(
                "torch_npu.profiler.analysis.prof_common_func._path_manager.os.listdir"
            ) as mock_listdir,
            patch(
                "torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path",
                return_value="/fake/cann/path",
            ),
            patch(
                "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.shutil.which",
                return_value="/usr/bin/msprof",
            ),
        ):
            mock_listdir.return_value = ["cann1", "cann2"]
            parser = CANNExportParser(
                "test", {"profiler_path": "/fake/path", Constant.EXPORT_TYPE: ["db"]}
            )
            self.assertEqual(parser._profiler_path, "/fake/path")
            self.assertEqual(parser._export_type, ["db"])
            self.assertIsNotNone(parser._cann_path)
            self.assertIsNotNone(parser.msprof_path)

    def test_cann_export_parser_msprof_path_none(self):
        with (
            patch("os.path.isdir", return_value=True),
            patch("subprocess.run") as mock_run,
            patch(
                "torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path",
                return_value="/fake/cann/path",
            ),
            patch(
                "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.shutil.which",
                return_value="/usr/bin/msprof",
            ),
            patch(
                "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.ProfilerConfig"
            ) as mock_config,  # noqa: F841
            patch(
                "torch_npu.profiler.analysis.prof_view.cann_parse._cann_export.ProfilerLogger"
            ) as mock_logger,  # noqa: F841
        ):
            mock_run.return_value.returncode = 0
            parser = CANNExportParser(
                "test", {"profiler_path": "/fake/path", Constant.EXPORT_TYPE: ["db"]}
            )
            parser.msprof_path = None

            result = parser.run({})
            self.assertEqual(result, (Constant.FAIL, None))


if __name__ == "__main__":
    run_tests()
