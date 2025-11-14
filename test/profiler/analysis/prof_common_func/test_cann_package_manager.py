import unittest
from unittest.mock import patch, Mock

from torch_npu.profiler.analysis.prof_common_func._cann_package_manager import (
    check_cann_package_support_export_db,
    check_cann_package_support_default_export_db,
    check_msprof_help_output,
    CannPackageManager
)


class TestCannPackageManager(unittest.TestCase):

    @patch('subprocess.run')
    @patch('torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.check_path_permission')
    @patch('shutil.which')
    def test_check_cann_package_support_export_db(self, mock_which, mock_check_permission, mock_run):
        mock_which.return_value = "/usr/bin/msprof"
        mock_check_permission.return_value = True
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.stdout = "--type option available"
        mock_run.return_value = mock_process

        result = check_cann_package_support_export_db()

        self.assertTrue(result)

    @patch('subprocess.run')
    @patch('torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.check_path_permission')
    @patch('shutil.which')
    def test_check_cann_package_support_default_export_db(self, mock_which, mock_check_permission, mock_run):
        mock_which.return_value = "/usr/bin/msprof"
        mock_check_permission.return_value = True
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.stdout = "default text: text(which will also export the database)"
        mock_run.return_value = mock_process

        result = check_cann_package_support_default_export_db()

        self.assertTrue(result)

    def test_cann_package_manager_cache_behavior(self):
        CannPackageManager.SUPPORT_EXPORT_DB = None
        CannPackageManager.SUPPORT_DEFAULT_EXPORT_DB = None

        with patch(
                'torch_npu.profiler.analysis.prof_common_func._cann_package_manager.check_cann_package_support_export_db') as mock_export, \
                patch(
                    'torch_npu.profiler.analysis.prof_common_func._cann_package_manager.check_cann_package_support_default_export_db') as mock_default:
            mock_export.return_value = True
            mock_default.return_value = True

            result1_export = CannPackageManager.is_support_export_db()
            result1_default = CannPackageManager.is_support_default_export_db()
            result2_export = CannPackageManager.is_support_export_db()
            result2_default = CannPackageManager.is_support_default_export_db()

            self.assertTrue(result1_export)
            self.assertTrue(result1_default)
            self.assertTrue(result2_export)
            self.assertTrue(result2_default)

            mock_export.assert_called_once()
            mock_default.assert_called_once()

    @patch('subprocess.run')
    @patch('torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.check_path_permission')
    @patch('shutil.which')
    def test_check_msprof_help_output_non_zero_exit(self, mock_which, mock_check_permission, mock_run):
        mock_which.return_value = '/usr/bin/msprof'
        mock_check_permission.return_value = True
        mock_process = Mock()
        mock_process.stdout = "some output"
        mock_run.return_value = mock_process

        result = check_msprof_help_output("test")
        self.assertFalse(result)

    @patch('torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.check_path_permission')
    @patch('shutil.which')
    def test_check_msprof_help_output_permission_denied(self, mock_which, mock_check_permission):
        mock_which.return_value = '/usr/bin/msprof'
        mock_check_permission.return_value = False

        result = check_msprof_help_output("test")
        self.assertFalse(result)

    @patch('torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.check_path_permission')
    @patch('shutil.which')
    def test_check_msprof_help_output_msprof_not_found(self, mock_which, mock_check_permission):
        mock_which.return_value = None
        mock_check_permission.return_value = True

        result = check_msprof_help_output("test")
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
