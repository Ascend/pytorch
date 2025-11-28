import unittest
from unittest.mock import patch, MagicMock
from torch_npu.profiler.analysis.prof_common_func._constant import Constant
from torch_npu.profiler.analysis.prof_view._kernel_view_parser import KernelViewParser


def run_test():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestKernelViewParser))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


class TestKernelViewParser(unittest.TestCase):

    def test_profect_map_for_headers_mixed(self):
        input_headers = ["Op Name", "Unknown Header", "Kernel Duration"]
        result = KernelViewParser._project_map_for_headers(input_headers)
        expected = {"op_name", "Unknown Header", "Kernel Duration"}
        self.assertEqual(result, expected)

    @patch('torch_npu.profiler.analysis.prof_view._kernel_view_parser.ProfilerConifg')
    @patch('torch_npu.profiler.analysis.prof_view._kernel_view_parser.CANNFileParser')
    @patch('torch_npu.profiler.analysis.prof_view._kernel_view_parser.FileManager')
    @patch('torch_npu.profiler.analysis.prof_view._kernel_view_parser.FwkCANNRelationParser')
    def test_run_success(self, mock_relation_parser, mock_file_manager, mock_cann_parser,
                          mock_config):
        mock_config_instance = mock_config.return_value
        mock_config_instance.load_info.return_value = None
        mock_cann_parser_instance = mock_cann_parser.return_value
        mock_cann_parser_instance.get_file_list_by_type.return_value = ["test_file.csv"]
        mock_file_manager.read_csv_file.return_value = [MagicMock(row=["test", "data"])]
        mock_relation_parser_instance = mock_relation_parser.return_value
        mock_relation_parser_instance.get_step_range.return_value = [{"step_id": 1, "start_ts": 1000, "end_ts": 2000}]

        parser = KernelViewParser("test", {})
        parser._profiler_path = "/test/path"
        parser._output_path = "/test/output"
        deps_data = {
            Constant.TREE_BUILD_PARSER: [MagicMock()],
            Constant.RELATION_PARSER: {"test": "data"}
        }
        result = parser.run(deps_data)
        self.assertEqual(result, (Constant.SUCCESS, None))

    def test_profect_map_for_headers_matching(self):
        input_headers = ["Name", "Kernel Time (ns)", "Calls"]
        result = KernelViewParser._project_map_for_headers(input_headers)
        expected = ["op_name", "kernel_time", "calls"]
        self.assertEqual(result, expected)


if __name__ == '__main__':
    run_test()