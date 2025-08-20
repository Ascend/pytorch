import unittest
from unittest.mock import patch, Mock

from torch_npu.profiler.analysis.prof_common_func._cann_package_manager import (
    check_cann_package_support_export_db,
    check_cann_package_support_default_export_db
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


if __name__ == '__main__':
    unittest.main()
