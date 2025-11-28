import unittest
from unittest.mock import patch
from torch_npu.profiler.analysis.prof_view._trace_view_parser import TraceViewParser


def run_test():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestTraceViewParser))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


class TestTraceViewParser(unittest.TestCase):

    def test_prune_trace_by_level_with_pruning(self):
        json_data = [
            {"name": "prune_me", "args": {"name": "other"}},
            {"name": "keep_me", "args": {"name": "keep_me"}},
        ]
        with patch('torch_npu.profiler.analysis.prof_view._trace_view_parser.ProfilerConfig') as mock_config:
            mock_config_instance = mock_config.return_value
            mock_config_instance.get_prune_config.return_value = {"prune_me": True}
            result = TraceViewParser._prune_trace_by_level(json_data)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["name"], "keep_me")

    def test_prune_trace_by_level_empty_data(self):
        result = TraceViewParser._prune_trace_by_level([])
        self.assertEqual(result, [])

    def test_trace_view_parser_init_with_directory_path(self):
        parser = TraceViewParser("test", {"output_path": "/test/output"})
        self.assertEqual(parser._trace_file_path, "test/output/trace_view.json")
        self.assertEqual(parser._temp_trace_file_path, "/test/output/trace_view.json")


if __name__ == "__main__":
    run_test()