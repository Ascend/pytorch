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
        param_dict = {"profiler_path": "/tmp", "export_type": ["db"]}
        with patch('shutil.which') as mock_which:
            mock_which.return_value = "/usr/bin/msprof"
            with patch('os.path.isdir') as mock_isdir:
                mock_isdir.return_value = True
                with patch('subprocess.run') as mock_run:
                    mock_run.return_value.returncode = 0
                    parser = CANNAnalyzeParser("test_parser", param_dict)
                    result = parser.run({})
                    self.assertEqual(result[0], Constant.SUCCESS)

    def test_cann_analyze_parser_run_no_cann_path(self):
        param_dict = {"profiler_path": "/tmp", "export_type": ["db", "text"]}
        with patch('torch_npu.profiler.analysis.prof_view.cann_parse._cann_analyze.ProfilerPathManager.get_cann_path') as mock_get_path:
            mock_get_path.return_value = "/nonexistent/path"
            with patch("os.path.isdir") as mock_isdir:
                mock_isdir.return_value = False
                parser = CANNAnalyzeParser("test_parser", param_dict)
                result = parser.run({})
                self.assertEqual(result[0], Constant.SUCCESS)

    def test_cann_analyze_parser_init(self):
        param_dict = {"profiler_path": "/tmp", "export_type": ["db", "text"]}
        parser = CANNAnalyzeParser("test_parser", param_dict)
        self.assertEqual(parser._name, "test_parser")
        self.assertEqual(parser._param_dict, param_dict)
        self.assertIsNotNone(parser._cann_path)
        self.assertIsNotNone(parser.msprof_path)


if __name__ == "__main__":
    run_test()