from unittest.mock import patch, MagicMock
from torch_npu.profiler.analysis.prof_parse._cann_file_parser import CANNFileParser, CANNDataEnum
from torch_npu.testing.testcase import TestCase, run_tests


class TestFwkFileParser(TestCase):

    def test_combine_acl_to_npu_with_amtching_events(self):
        with patch('torch_npu.profiler.analysis.prof_parse._cann_file_parser.ProfilerLogger') as mock_logger:
            mock_logger.get_instance.return_value = MagicMock()
            mock_logger.init.return_value = None

            timeline_data = [
                {"cat": "HostToDevice", "ph": "s", "id": 1, "ts": 1000},
                {"cat": "HostToDevice", "ph": "f", "id": 1, "ts": 2000, "pid": 1, "tid": 1},
                {"ph": "X", "pid": 1, "tid": 1, "ts": 2000, "name": "kernel_op"}
            ]
            result = CANNFileParser.combine_acl_to_npu(timeline_data)
            self.assertIsInstance(result, dict)
            self.assertIn(1000000, result)
            self.assertEqual(len(result[1000000]), 1)

    def test_json_dict_load_empty_data(self):
        result = CANNFileParser._json_dict_load("")
        self.assertEqual(result, {})

    def test_json_load_empty_data(self):
        result = CANNFileParser._json_load("")
        self.assertEqual(result, {})

    def test_get_timeline_all_data_empty(self):
        with patch('torch_npu.profiler.analysis.prof_parse._cann_file_parser.ProfilerLogger') as mock_logger, \
                patch(
                    'torch_npu.profiler.analysis.prof_parse._cann_file_parser.ProfilerPathManager') as mock_path_manager, \
                patch('torch_npu.profiler.analysis.prof_parse._cann_file_parser.FileManager') as mock_file_manager:
            mock_logger.get_instance.return_value = MagicMock()
            mock_path_manager.get_cann_path.return_value = "/test/path"
            mock_file_manager.check_file_readable.return_value = True
            mock_file_manager.check_file_writable.return_value = True
            mock_file_manager.file_read_all.return_value = ""

            parser = CANNFileParser("/test/path")
            parser._file_dict = {CANNDataEnum.MSPROF_TIMELINE: set()}

            result = parser.get_timeline_all_data()
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 0)

    def test_json_dict_load_invalid_json_raises_runtime_error(self):
        with self.assertRaises(RuntimeError):
            CANNFileParser._json_dict_load("{invalid json}")

    def test_json_dict_load_valid_dict(self):
        json_data = '{"key": "value", "key2": "value2"}'
        result = CANNFileParser._json_dict_load(json_data)
        self.assertEqual(result, {"key": "value", "key2": "value2"})

    def test_json_load_invalid_json_raises_runtime_error(self):
        with self.assertRaises(RuntimeError):
            CANNFileParser._json_load("{invalid json}")

    def test_json_load_valid_list(self):
        json_data = '[{"key": "value"}, {"key2": "value2"}]'
        result = CANNFileParser._json_load(json_data)
        self.assertEqual(result, [{"key": "value"}, {"key2": "value2"}])

    def test_json_dict_load_non_dict_data(self):
        result = CANNFileParser._json_dict_load('["key", "value"]')
        self.assertEqual(result, {})

    def test_json_load_non_list_data(self):
        result = CANNFileParser._json_load('{"key": "value"}')
        self.assertEqual(result, [])

    def test_get_acl_to_npu_data_with_matching_events(self):
        with patch('torch_npu.profiler.analysis.prof_parse._cann_file_parser.ProfilerLogger') as mock_logger, \
                patch(
                    'torch_npu.profiler.analysis.prof_parse._cann_file_parser.ProfilerPathManager') as mock_path_manager, \
                patch('torch_npu.profiler.analysis.prof_parse._cann_file_parser.FileManager') as mock_file_manager:
            mock_logger.get_instance.return_value = MagicMock()
            mock_logger.error.return_value = None
            mock_logger.warning.return_value = None
            mock_path_manager.get_cann_path.return_value = "/test/path"
            mock_file_manager.check_file_readable.return_value = True
            mock_file_manager.check_file_writable.return_value = True
            mock_file_manager.file_read_all.return_value = '[{"cat": "HostToDevice", "ph": "s", "id": 1, "ts": 1000}, {"cat": "HostToDevice", "ph": "f", "id": 1, "ts": 2000, "pid": 1, "tid": 1}, {"ph": "X", "pid": 1, "tid": 1, "ts": 2000, "name": "kernel_op"}]'

            parser = CANNFileParser("/test/path")
            parser._file_dict = {CANNDataEnum.MSPROF_TIMELINE: {"/test/timeline.json"}}

            result = parser.get_acl_to_npu_data()
            self.assertIsInstance(result, dict)
            self.assertIn(1000000, result)
            self.assertEqual(len(result[1000000]), 1)

    def test_get_analyze_communication_data_with_file(self):
        with patch('torch_npu.profiler.analysis.prof_parse._cann_file_parser.ProfilerLogger') as mock_logger, \
                patch(
                    'torch_npu.profiler.analysis.prof_parse._cann_file_parser.ProfilerPathManager') as mock_path_manager, \
                patch('torch_npu.profiler.analysis.prof_parse._cann_file_parser.FileManager') as mock_file_manager:
            mock_logger.get_instance.return_value = MagicMock()
            mock_path_manager.get_cann_path.return_value = "/test/path"
            mock_file_manager.check_file_readable.return_value = True
            mock_file_manager.check_file_writable.return_value = True
            mock_file_manager.file_read_all.return_value = '{"key": "value"}'

            parser = CANNFileParser("/test/path")
            parser._file_dict = {CANNDataEnum.COMMUNICATION: {"/test/path/communication.json"}}

            result = parser.get_analyze_communication_data(CANNDataEnum.COMMUNICATION)
            self.assertIsInstance(result, dict)
            self.assertEqual(result, {"key": "value"})

    def test_get_timeline_all_data_with_non_empty_timeline(self):
        with patch('torch_npu.profiler.analysis.prof_parse._cann_file_parser.ProfilerLogger') as mock_logger, \
                patch(
                    'torch_npu.profiler.analysis.prof_parse._cann_file_parser.ProfilerPathManager') as mock_path_manager, \
                patch('torch_npu.profiler.analysis.prof_parse._cann_file_parser.FileManager') as mock_file_manager:
            mock_logger.get_instance.return_value = MagicMock()
            mock_path_manager.get_cann_path.return_value = "/test/path"
            mock_file_manager.check_file_readable.return_value = True
            mock_file_manager.check_file_writable.return_value = True
            mock_file_manager.file_read_all.return_value = '[{"name": "test_event"}]'

            parser = CANNFileParser("/test/path")
            parser._file_dict = {CANNDataEnum.MSPROF_TIMELINE: {"/test/path/msprof_123.json"}}

            result = parser.get_timeline_all_data()
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["name"], "test_event")

    def test_combine_acl_to_npu_no_kernel_events(self):
        with patch('torch_npu.profiler.analysis.prof_parse._cann_file_parser.ProfilerLogger') as mock_logger:
            mock_logger.get_instance.return_value = MagicMock()
            mock_logger.error.return_value = None

            timeline_data = [
                {"cat": "HostToDevice", "ph": "s", "id": 1, "ts": 1000},
                {"cat": "HostToDevice", "ph": "s", "id": 1, "ts": 2000, "pid": 1, "tid": 1},
            ]
            result = CANNFileParser.combine_acl_to_npu(timeline_data)
            self.assertIsInstance(result, dict)
            self.assertEqual(len(result), 0)

    def test_combine_acl_to_npu_no_flow_events(self):
        with patch('torch_npu.profiler.analysis.prof_parse._cann_file_parser.ProfilerLogger') as mock_logger:
            mock_logger.get_instance.return_value = MagicMock()
            mock_logger.warning.return_value = None

            timeline_data = [
                {"ph": "X", "pid": 1, "tid": 1, "ts": 2000, "name": "kernel_op"},
            ]
            result = CANNFileParser.combine_acl_to_npu(timeline_data)
            self.assertIsInstance(result, dict)
            self.assertEqual(len(result), 0)


if __name__ == '__main__':
    run_tests()