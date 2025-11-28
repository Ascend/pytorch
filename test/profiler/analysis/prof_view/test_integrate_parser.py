import unittest
from unittest.mock import patch, MagicMock

from torch_npu.profiler.analysis.prof_parse._cann_file_parser import CANNDataEnum
from torch_npu.profiler.analysis.prof_view._integrate_parser import IntegrateParser


def run_test():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrateParser))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


class TestIntegrateParser(unittest.TestCase):

    def test_generate_view_with_multiple_parser_beans(self):
        with patch('torch_npu.profiler.analysis.prof_view._integrate_parser.ProfilerConfig') as mock_config_class:
            mock_config_instance = mock_config_class.return_value
            mock_config_instance.get_parser_bean.return_value = [
                (CANNDataEnum.NIC, "bean1"),
                (CANNDataEnum.ROCE, "bean2")
            ]
            mock_logger = MagicMock()
            with patch('torch_npu.profiler.analysis.prof_view._integrate_parser.ProfilerLogger') as mock_logger_class:
                mock_logger_class.get_instance.return_value = mock_logger
                with patch('torch_npu.profiler.analysis.prof_view._integrate_parser.IntegrateParser.generate_csv') as mock_generate_csv:
                    parser = IntegrateParser("test", {})
                    parser._output_path = "/fake/output/path"
                    parser._profiler_path = "/fake/profiler/path"
                    parser.generate_view()
                    self.assertEqual(mock_generate_csv.call_count, 2)

if __name__ == '__main__':
    run_test()