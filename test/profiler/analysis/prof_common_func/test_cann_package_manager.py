import os
import subprocess
import unittest
from unittest.mock import patch, Mock

from torch_npu.profiler.analysis.prof_common_func._cann_package_manager import (
    check_analysis_package_support_export_all,
    check_cann_package_support_default_export_db,
    check_cann_package_support_export_all,
    check_msprof_help_output,
    find_msprof_py_path,
    CannPackageManager
)


class TestCannPackageManager(unittest.TestCase):

    @patch('subprocess.run')
    @patch('shutil.which')
    def test_check_cann_package_support_default_export_db(self, mock_which, mock_run):
        mock_which.return_value = "/usr/bin/msprof"
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.stdout = "default text: text(which will also export the database)"
        mock_run.return_value = mock_process

        result = check_cann_package_support_default_export_db()

        self.assertTrue(result)

    @patch('subprocess.run')
    @patch('shutil.which')
    def test_check_cann_package_support_export_all(self, mock_which, mock_run):
        mock_which.return_value = "/usr/bin/msprof"
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.stdout = "or when collecting data, include text, db, all."
        mock_run.return_value = mock_process

        result = check_cann_package_support_export_all()

        self.assertTrue(result)

    @patch('subprocess.run')
    @patch('shutil.which')
    def test_check_cann_package_support_export_all_with_old_package(
        self, mock_which, mock_run
    ):
        mock_which.return_value = "/usr/bin/msprof"
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.stdout = "or when collecting data, include text, db."
        mock_run.return_value = mock_process

        result = check_cann_package_support_export_all()

        self.assertFalse(result)

    @patch('subprocess.run')
    def test_check_analysis_package_support_export_all(self, mock_run):
        mock_process = Mock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process

        self.assertTrue(
            check_analysis_package_support_export_all('/fake/tools/profiler/msprof.py')
        )
        self.assertEqual(
            mock_run.call_args[0][0],
            ["python3", "/fake/tools/profiler/msprof.py", "export", "all", "--help"],
        )

    @patch('subprocess.run')
    def test_check_analysis_package_support_export_all_with_old_package(self, mock_run):
        mock_process = Mock()
        mock_process.returncode = 2
        mock_run.return_value = mock_process

        self.assertFalse(
            check_analysis_package_support_export_all('/fake/tools/profiler/msprof.py')
        )

    @patch('subprocess.run')
    def test_check_analysis_package_support_export_all_without_path(self, mock_run):
        self.assertFalse(check_analysis_package_support_export_all(""))
        mock_run.assert_not_called()

    @patch('os.path.exists')
    @patch('shutil.which')
    def test_find_msprof_py_path(self, mock_which, mock_exists):
        mock_which.return_value = "/fake/ascend-toolkit/latest/tools/profiler/bin/msprof"
        mock_exists.return_value = True

        self.assertEqual(
            find_msprof_py_path(),
            os.path.join(
                "/fake/ascend-toolkit/latest/",
                "tools/profiler/profiler_tool/analysis/msprof/msprof.py",
            ),
        )

    @patch('shutil.which')
    def test_find_msprof_py_path_msprof_not_found(self, mock_which):
        mock_which.return_value = None

        self.assertEqual(find_msprof_py_path(), "")

    def test_cann_package_manager_cache_behavior(self):
        CannPackageManager.SUPPORT_DEFAULT_EXPORT_DB = None
        CannPackageManager.SUPPORT_EXPORT_ALL = None

        with patch(
            'torch_npu.profiler.analysis.prof_common_func._cann_package_manager.'
            'check_cann_package_support_default_export_db'
        ) as mock_default, patch(
            'torch_npu.profiler.analysis.prof_common_func._cann_package_manager.'
            'check_cann_package_support_export_all'
        ) as mock_all, patch(
            'torch_npu.profiler.analysis.prof_common_func._cann_package_manager.'
            'find_msprof_py_path'
        ) as mock_find, patch(
            'torch_npu.profiler.analysis.prof_common_func._cann_package_manager.'
            'check_analysis_package_support_export_all'
        ) as mock_analysis_all:
            mock_default.return_value = True
            mock_all.return_value = True
            mock_find.return_value = "/fake/tools/profiler/msprof.py"
            mock_analysis_all.return_value = True

            result1_default = CannPackageManager.is_support_default_export_db()
            result1_all = CannPackageManager.is_support_export_all()
            result2_default = CannPackageManager.is_support_default_export_db()
            result2_all = CannPackageManager.is_support_export_all()

            self.assertTrue(result1_default)
            self.assertTrue(result1_all)
            self.assertTrue(result2_default)
            self.assertTrue(result2_all)

            mock_default.assert_called_once()
            mock_all.assert_called_once()
            mock_find.assert_called_once()
            mock_analysis_all.assert_called_once()

    def test_cann_package_manager_export_all_requires_analysis_package(self):
        CannPackageManager.SUPPORT_EXPORT_ALL = None

        with patch(
            'torch_npu.profiler.analysis.prof_common_func._cann_package_manager.'
            'check_cann_package_support_export_all',
            return_value=True,
        ), patch(
            'torch_npu.profiler.analysis.prof_common_func._cann_package_manager.'
            'find_msprof_py_path',
            return_value="/fake/tools/profiler/msprof.py",
        ), patch(
            'torch_npu.profiler.analysis.prof_common_func._cann_package_manager.'
            'check_analysis_package_support_export_all',
            return_value=False,
        ):
            self.assertFalse(CannPackageManager.is_support_export_all())
        CannPackageManager.SUPPORT_EXPORT_ALL = None

    @patch('subprocess.run')
    @patch('shutil.which')
    def test_check_msprof_help_output_timeout(self, mock_which, mock_run):
        mock_which.return_value = "/usr/bin/msprof"
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="msprof", timeout=30)

        self.assertFalse(check_msprof_help_output("--type", 30))

    @patch('subprocess.run')
    def test_check_analysis_package_support_export_all_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="python3", timeout=60)

        self.assertFalse(
            check_analysis_package_support_export_all('/fake/tools/profiler/msprof.py')
        )

    @patch('subprocess.run')
    @patch('shutil.which')
    def test_check_msprof_help_output_non_zero_exit(self, mock_which, mock_run):
        mock_which.return_value = '/usr/bin/msprof'
        mock_process = Mock()
        mock_process.stdout = "some output"
        mock_run.return_value = mock_process

        result = check_msprof_help_output("test")
        self.assertFalse(result)

    @patch('shutil.which')
    def test_check_msprof_help_output_msprof_not_found(self, mock_which):
        mock_which.return_value = None

        result = check_msprof_help_output("test")
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
