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
from unittest.mock import patch

from torch_npu.profiler.analysis._profiling_parser import ProfilingParser
from torch_npu.profiler.analysis.prof_common_func._constant import Constant
from torch_npu.testing.testcase import run_tests, TestCase


class TestProfilingParser(TestCase):
    def _build_prof_dir(self, profiler_path: str):
        cann_path = os.path.join(profiler_path, "PROF_000001_20260101000000000_ABCDEF")
        device_path = os.path.join(cann_path, "device_0")
        host_path = os.path.join(cann_path, "host")
        for dir_name in ["analyze", "mindstudio_profiler_log", "mindstudio_profiler_output"]:
            os.makedirs(os.path.join(cann_path, dir_name))
        for base_path in [device_path, host_path]:
            os.makedirs(base_path)
            for dir_name in ["sqlite", "summary", "timeline"]:
                os.makedirs(os.path.join(base_path, dir_name))
        return cann_path, device_path, host_path

    def _simplify_data(self, profiler_path: str, cann_path: str, device_path: str,
                       host_path: str, simplify_flag: bool):
        with (
            patch(
                "torch_npu.profiler.analysis._profiling_parser.ProfilerPathManager.get_cann_path",
                return_value=cann_path,
            ),
            patch(
                "torch_npu.profiler.analysis._profiling_parser.ProfilerPathManager.get_device_path",
                return_value=[device_path],
            ),
            patch(
                "torch_npu.profiler.analysis._profiling_parser.ProfilerPathManager.get_host_path",
                return_value=host_path,
            ),
        ):
            ProfilingParser.simplify_data(profiler_path, simplify_flag)

    def test_simplify_data_keeps_mindstudio_profiler_log(self):
        with tempfile.TemporaryDirectory() as profiler_path:
            cann_path, device_path, host_path = self._build_prof_dir(profiler_path)
            analysis_log = os.path.join(cann_path, "msprof_analysis_1234.log")
            with open(analysis_log, "w") as file:
                file.write("log")

            self._simplify_data(profiler_path, cann_path, device_path, host_path, True)

            self.assertTrue(
                os.path.isdir(os.path.join(cann_path, "mindstudio_profiler_log"))
            )
            self.assertFalse(os.path.exists(os.path.join(cann_path, "analyze")))
            self.assertFalse(
                os.path.exists(os.path.join(cann_path, "mindstudio_profiler_output"))
            )
            for base_path in [device_path, host_path]:
                for dir_name in ["sqlite", "summary", "timeline"]:
                    self.assertFalse(os.path.exists(os.path.join(base_path, dir_name)))
            self.assertTrue(os.path.exists(analysis_log))

    def test_remove_cann_output_removes_previous_artifacts(self):
        with tempfile.TemporaryDirectory() as profiler_path:
            cann_path, device_path, host_path = self._build_prof_dir(profiler_path)
            logs_dir = os.path.join(cann_path, "mindstudio_profiler_log")
            analysis_log = os.path.join(cann_path, "msprof_analysis_1234.log")
            with open(analysis_log, "w") as file:
                file.write("log")

            ProfilingParser.remove_cann_output(cann_path, True)

            for base_path in [device_path, host_path]:
                for dir_name in ["sqlite", "summary", "timeline"]:
                    self.assertFalse(os.path.exists(os.path.join(base_path, dir_name)))
            self.assertFalse(os.path.exists(os.path.join(cann_path, "analyze")))
            self.assertFalse(
                os.path.exists(os.path.join(cann_path, "mindstudio_profiler_output"))
            )
            self.assertTrue(os.path.isdir(logs_dir))
            self.assertTrue(os.path.exists(analysis_log))

    def test_remove_cann_output_only_removes_sqlite_when_not_all(self):
        with tempfile.TemporaryDirectory() as profiler_path:
            cann_path, device_path, host_path = self._build_prof_dir(profiler_path)

            ProfilingParser.remove_cann_output(cann_path, False)

            for base_path in [device_path, host_path]:
                self.assertFalse(os.path.exists(os.path.join(base_path, "sqlite")))
                self.assertTrue(os.path.exists(os.path.join(base_path, "summary")))
                self.assertTrue(os.path.exists(os.path.join(base_path, "timeline")))
            self.assertTrue(os.path.isdir(os.path.join(cann_path, "analyze")))
            self.assertTrue(
                os.path.isdir(os.path.join(cann_path, "mindstudio_profiler_output"))
            )

    def test_run_parser_cleans_cann_output_before_parse(self):
        with tempfile.TemporaryDirectory() as profiler_path:
            cann_path, _, _ = self._build_prof_dir(profiler_path)
            parser = ProfilingParser(
                profiler_path,
                Constant.TENSORBOARD_TRACE_HANDLER,
                profiler_path,
                {},
            )
            with (
                patch(
                    "torch_npu.profiler.analysis._profiling_parser.ProfilerPathManager.get_cann_path",
                    return_value=cann_path,
                ),
                patch(
                    "torch_npu.profiler.analysis._profiling_parser.ProfilingParser.remove_cann_output"
                ) as mock_remove,
                patch(
                    "torch_npu.profiler.analysis._profiling_parser.ProfilerConfig"
                ) as mock_config,
                patch(
                    "torch_npu.profiler.analysis._profiling_parser.ParserConfig"
                ) as mock_parser_config,
                patch(
                    "torch_npu.profiler.analysis._profiling_parser.CannPackageManager.is_support_default_export_db",
                    return_value=False,
                ),
                # 用例只关心清理发生的时机，不真正启动解析任务，避免依赖 ProfilerLogger 初始化
                patch(
                    "torch_npu.profiler.analysis._profiling_parser.ConcurrentTasksManager"
                ) as mock_manager,
            ):
                mock_config.return_value.export_type = [Constant.Text]
                mock_config.return_value.get_level.return_value = Constant.LEVEL1
                mock_parser_config.ONLY_FWK_CONFIG = {}
                mock_parser_config.COMMON_CONFIG = {}
                parser.run_parser()
            mock_remove.assert_called_once_with(cann_path, True)
            mock_manager.return_value.run.assert_called_once()

    def test_update_export_type_db_no_longer_requires_cann_check(self):
        with tempfile.TemporaryDirectory() as profiler_path:
            cann_path, _, _ = self._build_prof_dir(profiler_path)
            parser = ProfilingParser(
                profiler_path,
                Constant.TENSORBOARD_TRACE_HANDLER,
                profiler_path,
                {"export_type": [Constant.Db]},
            )
            with (
                patch(
                    "torch_npu.profiler.analysis._profiling_parser.ProfilerConfig"
                ) as mock_config,
                patch(
                    "torch_npu.profiler.analysis._profiling_parser.ProfilerPathManager.get_cann_path",
                    return_value=cann_path,
                ),
            ):
                mock_config.return_value.export_type = [Constant.Db]
                # 不再依赖 CANN 包的 export db 能力判定，不应抛错
                parser.update_export_type()

    def test_simplify_data_without_simplify_flag_keeps_summary(self):
        with tempfile.TemporaryDirectory() as profiler_path:
            cann_path, device_path, host_path = self._build_prof_dir(profiler_path)

            self._simplify_data(profiler_path, cann_path, device_path, host_path, False)

            self.assertTrue(
                os.path.isdir(os.path.join(cann_path, "mindstudio_profiler_log"))
            )
            self.assertTrue(os.path.isdir(os.path.join(cann_path, "analyze")))
            for base_path in [device_path, host_path]:
                self.assertFalse(os.path.exists(os.path.join(base_path, "sqlite")))
                self.assertTrue(os.path.exists(os.path.join(base_path, "summary")))
                self.assertTrue(os.path.exists(os.path.join(base_path, "timeline")))


if __name__ == "__main__":
    run_tests()
