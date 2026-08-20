import tempfile
from unittest.mock import patch

from torch_npu.profiler.analysis.prof_bean._op_summary_bean import OpSummaryBean
from torch_npu.profiler.analysis.prof_common_func._csv_headers import CsvHeaders
from torch_npu.profiler.analysis.prof_view._kernel_view_parser import KernelViewParser
from torch_npu.testing.testcase import TestCase, run_tests


class TestKernelViewParser(TestCase):

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.parser_params = {
            "profiler_path": self.temp_dir.name,
            "output_path": self.temp_dir.name,
        }
        OpSummaryBean.headers = []

    def tearDown(self):
        OpSummaryBean.headers = []
        self.temp_dir.cleanup()
        super().tearDown()

    def test_project_map_for_headers(self):
        input_headers = ["Op Name", "Unknown Header", "Task Duration(us)"]

        result = KernelViewParser._project_map_for_headers(input_headers)

        self.assertEqual(["Name", "Unknown Header", "Duration(us)"], result)

    def test_get_kernel_headers_for_level0_without_shape(self):
        all_headers = CsvHeaders.OP_SUMMARY_SHOW_HEADERS + ["Model ID"]

        result = KernelViewParser._get_kernel_headers(all_headers, False)

        self.assertEqual(CsvHeaders.OP_SUMMARY_SHOW_HEADERS, result)

    def test_get_kernel_headers_for_level0_with_shape(self):
        all_headers = (CsvHeaders.OP_SUMMARY_SHOW_HEADERS + ["Model ID"]
                       + CsvHeaders.OP_SUMMARY_SHAPE_HEADERS)

        result = KernelViewParser._get_kernel_headers(all_headers, False)

        self.assertEqual(CsvHeaders.OP_SUMMARY_SHOW_HEADERS + CsvHeaders.OP_SUMMARY_SHAPE_HEADERS, result)

    def test_get_kernel_headers_for_non_level0(self):
        all_headers = CsvHeaders.OP_SUMMARY_SHOW_HEADERS + ["Model ID"]

        result = KernelViewParser._get_kernel_headers(all_headers, True)

        self.assertEqual(all_headers, result)

    @patch("torch_npu.profiler.analysis.prof_view._kernel_view_parser.ProfilerConfig")
    @patch("torch_npu.profiler.analysis.prof_view._kernel_view_parser.CANNFileParser")
    @patch("torch_npu.profiler.analysis.prof_view._kernel_view_parser.FileManager")
    def test_generate_level0_view_without_shape(self, mock_file_manager, mock_cann_parser, mock_config):
        source_headers = CsvHeaders.OP_SUMMARY_SHOW_HEADERS + ["Model ID"]
        source_data = {header: str(index) for index, header in enumerate(source_headers)}
        mock_cann_parser.return_value.get_file_list_by_type.return_value = ["op_summary.csv"]
        mock_file_manager.read_csv_file.return_value = [OpSummaryBean(source_data)]
        mock_config.return_value.is_all_kernel_headers.return_value = False
        parser = KernelViewParser("test", self.parser_params)

        parser.generate_view()

        expected_row = [[source_data.get(header) for header in CsvHeaders.OP_SUMMARY_SHOW_HEADERS]]
        mock_file_manager.create_csv_file.assert_called_once_with(
            parser._output_path, expected_row, parser.KERNEL_VIEW, CsvHeaders.OP_SUMMARY_KERNEL_BASE_HEADERS)

    @patch("torch_npu.profiler.analysis.prof_view._kernel_view_parser.ProfilerConfig")
    @patch("torch_npu.profiler.analysis.prof_view._kernel_view_parser.CANNFileParser")
    @patch("torch_npu.profiler.analysis.prof_view._kernel_view_parser.FileManager")
    def test_generate_level0_view_with_shape(self, mock_file_manager, mock_cann_parser, mock_config):
        source_headers = (CsvHeaders.OP_SUMMARY_SHOW_HEADERS + ["Model ID"]
                          + CsvHeaders.OP_SUMMARY_SHAPE_HEADERS)
        source_data = {header: str(index) for index, header in enumerate(source_headers)}
        mock_cann_parser.return_value.get_file_list_by_type.return_value = ["op_summary.csv"]
        mock_file_manager.read_csv_file.return_value = [OpSummaryBean(source_data)]
        mock_config.return_value.is_all_kernel_headers.return_value = False
        parser = KernelViewParser("test", self.parser_params)

        parser.generate_view()

        expected_source_headers = CsvHeaders.OP_SUMMARY_SHOW_HEADERS + CsvHeaders.OP_SUMMARY_SHAPE_HEADERS
        expected_row = [[source_data.get(header) for header in expected_source_headers]]
        expected_output_headers = CsvHeaders.OP_SUMMARY_KERNEL_BASE_HEADERS + CsvHeaders.OP_SUMMARY_SHAPE_HEADERS
        mock_file_manager.create_csv_file.assert_called_once_with(
            parser._output_path, expected_row, parser.KERNEL_VIEW, expected_output_headers)


if __name__ == "__main__":
    run_tests()
