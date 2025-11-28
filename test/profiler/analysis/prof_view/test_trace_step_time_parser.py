import unittest
from collections import defaultdict
from unittest.mock import patch, MagicMock
from torch_npu.profiler.analysis.prof_view._trace_step_time_parser import default_time, step_time_dict, TraceStepTimeParser


def run_test():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestTraceStepTimeParser))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


class TestTraceStepTimeParser(unittest.TestCase):

    def etst_get_prepare_time_valid(self):
        step = 1
        step_list = [[1, 100, 1000, 200, 900, 150, 300]]

        with patch('torch_npu.profiler.analysis.prof_view._trace_step_time_parser.FwFileParser') as mock_fwk_parser:
            mock_instance = mock_fwk_parser.return_value
            mock_instance.get_first_fwk_op.return_value = MagicMock(ts=100)

            parser = TraceStepTimeParser("test", {})
            result = parser.get_prepare_time(step, step_list)
            self.assertEqual(result, 200)

    def test_get_e2e_time_valid(self):
        step = 1
        step_list = [[1, 100, 1000, 200, 900, -1, -1]]

        result = TraceStepTimeParser.get_e2e_time(step, step_list)
        self.assertEqual(result, 700)

    def test_is_float_num_method(self):
        self.assertTrue(TraceStepTimeParser.is_float_num("123.45"))
        self.assertTrue(TraceStepTimeParser.is_float_num("123"))
        self.assertTrue(TraceStepTimeParser.is_float_num("-123.45"))
        self.assertTrue(TraceStepTimeParser.is_float_num("0"))

        self.assertFalse(TraceStepTimeParser.is_float_num("abc"))
        self.assertFalse(TraceStepTimeParser.is_float_num(""))
        self.assertFalse(TraceStepTimeParser.is_float_num("12.34.56"))

    def test_step_time_dict_function(self):
        result = step_time_dict()
        self.assertIsInstance(result, defaultdict)
        self.assertEqual(result.default_factory, default_time)
        result["test_key"]["compute"] = 100
        self.assertEqual(result["test_key"]["compute"], 100)

    def test_default_time_function(self):
        result = default_time()
        expected_keys = ["compute", "comunNotOverlp", "Overlp", "comun", "free", "stage", "bubble", "comunNotOverLpRec", "prepare"]
        self.assertEqual(list(result.keys()), expected_keys)
        for value in result.values():
            self.assertEqual(value, 0)


if __name__ == "__main__":
    run_test()