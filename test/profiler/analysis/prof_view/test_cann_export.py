import unittest
from unittest.mock import patch
from torch_npu.profiler.analysis.prof_common_func._constant import Constant
from torch_npu.profiler.analysis.prof_view.cann_parse._cann_export import CANNExportParser


def run_test():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestCannExport))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


class TestCannExport(unittest.TestCase):
    def test_cann_export_parser_run_db_export_success(self):
        with patch("os.patch.isdir", return_value=True), \
                patch("subprocess.run") as mock_run, \
                patch('from torch_npu.profiler.analysis.prof_common_func._path_manager.ProfilerPathManager.get_cann_path', return_value="/fake/cann/path"):
            mock_run.return_value.returncode = 0
            parser = CANNExportParser("test", {"profiler_path": "/fake/path", "export_type": ["db"]})
            parser.msprof_path = "/usr/bin/msprof"
            result = parser.run({})
            self.assertEqual(result, (Constant.SUCCESS, None))

    def test_cann_export_parser_init(self):
        parser = CANNExportParser("test", {"profiler_path": "/fake/path", "export_type": ["db"]})
        self.assertEqual(parser._profiler_path, "/fake/path")
        self.assertEqual(parser._export_type, "db")
        self.assertIsNotNone(parser._cann_path)
        self.assertIsNotNone(parser.msprof_path)


if __name__ == "__main__":
    run_test()