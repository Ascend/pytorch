import os
import fcntl
import shutil
import traceback
from unittest.mock import patch
import torch_npu.utils._npu_trace as npu_trace
from torch_npu.utils.utils import _print_info_log, _print_error_log, _print_warn_log
import torch_npu.npu._kernel_check as kernel_check
from torch_npu.testing.testcase import TestCase, run_tests


class TestKernelCheck(TestCase):
    def test_clear_debug_env(self):
        with patch.dict(os.environ, {
            "ASCEND_OPP_PATH": "/valid/opp/path",
            "ASCEND_OPP_DEBUG_PATH": "/valid/debug/path"}):
            manager = kernel_check.KernelPathManager()
            with patch('os.path.exists') as mock_exists:
                mock_exists.return_value = True
                with patch.object(manager, 'func_with_lock') as mock_func_with_lock:
                    with patch('os.remove') as mock_remove:
                        manager.clear_debug_env()
                        mock_func_with_lock.assert_called_once()
                        mock_remove.assert_called_once()

    def test_clear_debug_env_path_not_exists(self):
        with patch.dict(os.environ, {
            "ASCEND_OPP_PATH": "/valid/opp/path",
            "ASCEND_OPP_DEBUG_PATH": "/invalid/debug/path"}):
            manager = kernel_check.KernelPathManager()
            with patch('os.path.exists') as mock_exists:
                mock_exists.return_value = False
                with patch.object(manager, 'func_with_lock') as mock_func_with_lock:
                    manager.clear_debug_env()
                    mock_func_with_lock.assert_not_called()

    def test_removed_debug_files(self):
        with patch.dict(os.environ, {
            "ASCEND_OPP_PATH": "/valid/opp/path",
            "ASCEND_OPP_DEBUG_PATH": "/valid/debug/path"}):
            manager = kernel_check.KernelPathManager()
            with patch('os.path.exists') as mock_exists:
                with patch('os.unlink') as mock_unlink:
                    with patch('shutil.rmtree') as mock_rmtree:
                        mock_exists.return_value = True
                        manager.remove_debug_files()
                        mock_unlink.assert_called_once()
                        mock_rmtree.assert_called_once()

    def test_kernel_path_manager_init_valid_env(self):
        with patch.dict(os.environ, {
            'ASCEND_OPP_PATH': '/valid/opp/path',
            'ASCEND_OPP_DEBUG_PATH': '/valid/debug/path'
        }):
            with patch.object(kernel_check.KernelPathManager, 'make_opp_debug_path') as mock_make:
                manager = kernel_check.KernelPathManager()
                self.assertEqual(manager.ascend_opp_path, '/valid/opp/path')
                self.assertEqual(manager.opp_debug_kernel_path, '/valid/debug/path')
                self.assertEqual(os.environ['ASCEND_LAUNCH_BLOCKING'], '1')
                self.assertEqual(os.environ['ASCEND_OPP_PATH'], manager.opp_debug_path)
                mock_make.assert_called_once()

    def test_handle_acl_start_execution(self):
        handler = kernel_check.EventHandler()
        with patch.object(npu_trace, 'print_check_msg') as mock_print:
            handler._handle_acl_start_execution('test_acl')
            mock_print.assert_called_once_with("====== Start acl operator test_acl")

    def test_clear_debug_env_file_not_found(self):
        with patch.dict(os.environ, {
            'ASCEND_OPP_PATH': '/valid/opp/path',
            'ASCEND_OPP_DEBUG_PATH': '/valid/debug/path'
        }):
            manager = kernel_check.KernelPathManager()
            with patch('os.path.exists') as mock_exists:
                mock_exists.return_value = True
                with patch.object(manager, 'func_with_lock') as mock_func_with_lock:
                    with patch('os.remove') as mock_remove:
                        mock_remove.side_effect = FileNotFoundError("No such file or directory")
                        with patch.object(kernel_check, '_print_info_log') as mock_print_info:
                            manager.clear_debug_env()
                            mock_func_with_lock.assert_called_once()
                            mock_remove.assert_called_once()
                            mock_print_info.assert_called_once()

    def test_make_opp_debug_path_invalid_kernel_path(self):
        with patch.dict(os.environ, {
            'ASCEND_OPP_PATH': '/valid/opp/path',
            'ASCEND_OPP_DEBUG_PATH': '/valid/debug/path'
        }):
            manager = kernel_check.KernelPathManager()

            def mock_exists_side_effect(path):
                return path == '/valid/opp/path' or path == '/valid/debug/path'

            with patch('os.path.exists') as mock_exists:
                mock_exists.side_effect = mock_exists_side_effect
                with patch.object(kernel_check, '_print_warn_log') as mock_print_warn:
                    manager.make_opp_debug_path()
                    mock_print_warn.assert_called_once_with("ASCEND_OPP_DEBUG_PATH is not valid kernel path.")

    def test_make_opp_debug_path_debug_path_not_exists(self):
        with patch.dict(os.environ, {
            'ASCEND_OPP_PATH': '/valid/opp/path',
            'ASCEND_OPP_DEBUG_PATH': '/valid/debug/path'
        }):
            manager = kernel_check.KernelPathManager()

            def mock_exists_side_effect(path):
                return path == '/valid/opp/path'

            with patch('os.path.exists') as mock_exists:
                mock_exists.side_effect = mock_exists_side_effect
                with patch.object(kernel_check, '_print_error_log') as mock_print_error:
                    manager.make_opp_debug_path()
                    mock_print_error.assert_called_once_with("ASCEND_OPP_DEBUG_PATH is not exists.")


if __name__ == "__main__":
    run_tests()

