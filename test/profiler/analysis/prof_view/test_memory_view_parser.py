import unittest
from unittest.mock import patch, MagicMock
from torch_npu.profiler.analysis.prof_common_func._constant import Constant
from torch_npu.profiler.analysis.prof_view._memory_view_parser import MemoryViewParser


def run_test():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryViewParser))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


class TestMemoryViewParser(unittest.TestCase):

    def test_run_with_exception(self):
        parser = MemoryViewParser("test", {})
        parser._profiler_path = "/fake/path"
        parser.logger = MagicMock()
        with patch('torch_npu.profiler.analysis.prof_view._memory_view_parser.ProfilerPathManager.get_cann_path',
                   side_effect=Exception("Test error")):
            result = parser.run({})
            self.assertEqual(result, (Constant.FAIL, None))

    def test_combine_record_workspace_type(self):
        mock_record = MagicMock()
        mock_record.component_type = "workspace"
        mock_record.time_ns = 1000
        mock_record.total_allocated = 100
        mock_record.total_reserved = 200
        mock_record.total_active = 300
        mock_record.stream_ptr = "stream1"
        mock_record.device_tag = "device1"
        result = MemoryViewParser._combine_record({}, mock_record)
        expected = [["workspace", "0.001000\t", 100, 200, 300, "stream1", "device1"]]
        self.assertEqual(result, expected)

    def test_get_data_from_file_with_data(self):
        mock_file_set = {"fake_file.csv"}
        mock_bean = MagicMock()
        mock_bean.row = ["test", "data"]
        with patch('torch_npu.profiler.analysis.prof_view._memory_view_parser.FileManager.read_csv_file',
                   return_value=[mock_bean]):
            result = MemoryViewParser._get_data_from_file(mock_file_set, mock_bean, True)
            self.assertEqual(result, [mock_bean])

    def test_get_data_from_file_empty_set(self):
        result = MemoryViewParser._get_data_from_file(set(), None, False)
        self.assertEqual(result, [])


if __name__ == '__main__':
    run_test()