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

from unittest.mock import patch

from torch_npu.profiler.analysis.prof_common_func._constant import Constant
from torch_npu.profiler.analysis.prof_view.cann_parse._cann_export import (
    CANNExportParser,
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
        ):
            mock_run.return_value.returncode = 0
            parser = CANNExportParser(
                "test", {"profiler_path": "/fake/path", Constant.EXPORT_TYPE: ["db"]}
            )
            parser.msprof_path = "/usr/bin/msprof"
            result = parser.run({})
            self.assertEqual(result, (Constant.SUCCESS, None))

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
