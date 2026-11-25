import os
import unittest
from unittest.mock import patch
from torch_npu.profiler.analysis.prof_common_func._constant import Constant
from torch_npu.profiler.analysis.prof_view.cann_parse._cann_analyze import CANNAnalyzeParser


def run_test():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestCannAnalyze))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


class TestCannAnalyze(unittest.TestCase):

    def test_cann_analyze_parser_run_exception(self):
        param_dict = {"profiler_path": "/tmp", "export_type": ["db"]}
        with patch('torch_npu.profiler.analysis.prof_view.cann_parse._cann_analyze.ProfilerConfig') as mock_config:
            mock_config.side_effect = Exception("Test exception")
            parser = CANNAnalyzeParser("test_parser", param_dict)
            with patch('torch_npu.profiler.analysis.prof_view.cann_parse._cann_analyze.print_error_msg') as mock_error:
                result = parser.run({})
                self.assertEqual(result[0], Constant.FAIL)
                mock_error.assert_called_with("Failed to analyze CANN Profiling data.")

    def test_cann_analyze_parser_run_db_success(self):
        param_dict = {"profiler_path": "/tmp", Constant.EXPORT_TYPE: [Constant.Db]}
        with patch('shutil.which') as mock_which:
            mock_which.return_value = "/usr/bin/msprof"
            with patch('os.path.isdir') as mock_isdir:
                mock_isdir.return_value = True
                with patch('subprocess.run') as mock_run, patch(
                    'torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path',
                    return_value="/tmp",
                ), patch(
                    'torch_npu.profiler.analysis.prof_view.cann_parse._cann_analyze.ProfilerConfig'
                ), patch(
                    'torch_npu.profiler.analysis.prof_view.cann_parse._cann_analyze.save_command_output'
                ) as mock_save:
                    mock_run.return_value.returncode = 0
                    parser = CANNAnalyzeParser("test_parser", param_dict)
                    result = parser.run({})
                    self.assertEqual(result[0], Constant.SUCCESS)
                    mock_save.assert_called_once()

    def test_cann_analyze_parser_run_no_cann_path(self):
        param_dict = {"profiler_path": "/tmp", "export_type": ["db", "text"]}
        with patch(
            'torch_npu.profiler.analysis.prof_view.cann_parse._cann_analyze.'
            'ProfilerPathManager.get_cann_path'
        ) as mock_get_path:
            mock_get_path.return_value = "/nonexistent/path"
            with patch("os.path.isdir") as mock_isdir:
                mock_isdir.return_value = False
                parser = CANNAnalyzeParser("test_parser", param_dict)
                result = parser.run({})
                self.assertEqual(result[0], Constant.SUCCESS)

    def test_cann_analyze_parser_run_without_msprof(self):
        param_dict = {"profiler_path": "/tmp", "export_type": ["db"]}
        with patch('shutil.which', return_value=None), patch('os.path.isdir', return_value=True), patch(
            'torch_npu.profiler.analysis.prof_view.cann_parse._cann_analyze.print_error_msg'
        ):
            parser = CANNAnalyzeParser("test_parser", param_dict)
            result = parser.run({})
            self.assertEqual(result[0], Constant.FAIL)

    def test_cann_analyze_parser_saves_command_output(self):
        param_dict = {"profiler_path": "/tmp", Constant.EXPORT_TYPE: [Constant.Db, Constant.Text]}
        with patch('shutil.which', return_value="/usr/bin/msprof"), patch(
            'os.path.isdir', return_value=True
        ), patch(
            'torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path',
            return_value="/tmp",
        ), patch(
            'torch_npu.profiler.analysis.prof_view.cann_parse._cann_analyze.ProfilerConfig'
        ), patch('subprocess.run') as mock_run, patch(
            'torch_npu.profiler.analysis.prof_view.cann_parse._cann_analyze.save_command_output'
        ) as mock_save:
            mock_run.return_value.returncode = 0
            parser = CANNAnalyzeParser("test_parser", param_dict)
            result = parser.run({})
        self.assertEqual(result[0], Constant.SUCCESS)
        self.assertEqual(mock_save.call_count, 2)
        log_path = mock_save.call_args_list[0][0][0]
        self.assertIn("msprof_analyze_", os.path.basename(log_path))

    def test_cann_analyze_parser_run_command_failed(self):
        param_dict = {"profiler_path": "/tmp", Constant.EXPORT_TYPE: [Constant.Db, Constant.Text]}
        with patch('shutil.which', return_value="/usr/bin/msprof"), patch(
            'os.path.isdir', return_value=True
        ), patch(
            'torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path',
            return_value="/tmp",
        ), patch(
            'torch_npu.profiler.analysis.prof_view.cann_parse._cann_analyze.ProfilerConfig'
        ), patch('subprocess.run') as mock_run, patch(
            'torch_npu.profiler.analysis.prof_view.cann_parse._cann_analyze.save_command_output'
        ), patch(
            'torch_npu.profiler.analysis.prof_view.cann_parse._cann_analyze.print_warn_msg'
        ) as mock_warn:
            mock_run.return_value.returncode = 1
            parser = CANNAnalyzeParser("test_parser", param_dict)
            result = parser.run({})
        self.assertEqual(result[0], Constant.SUCCESS)
        self.assertEqual(mock_warn.call_count, 2)

    def test_cann_analyze_parser_init(self):
        param_dict = {"profiler_path": "/tmp", "export_type": ["db", "text"]}
        with patch("shutil.which", return_value="/usr/bin/msprof"):
            parser = CANNAnalyzeParser("test_parser", param_dict)
        self.assertEqual(parser.name, "test_parser")
        self.assertEqual(parser._param_dict, param_dict)
        self.assertIsNotNone(parser._cann_path)
        self.assertIsNotNone(parser.msprof_path)


if __name__ == "__main__":
    run_test()
