from unittest.mock import patch, MagicMock
from torch_npu.testing.testcase import TestCase, run_tests

from torch_npu.profiler.analysis.prof_parse._event_tree_parser import (
    build_event_tree,
    _EventType,
    parse_tensor_metadata,
    parse_input_from_string,
    mark_finished,
    push_event
)


class TestEventTreeParser(TestCase):

    def test_build_event_tree_error_conditions(self):
        mock_event = MagicMock()
        mock_event.finished = True
        mock_event.tid = 1
        mock_event.start_time_ns = 1000
        mock_event.end_time_ns = 1000
        mock_event.parent = None
        mock_event.children = []
        mock_event.tag = _EventType.TorchOp
        mock_event.extra_fields = MagicMock()
        mock_event.extra_fields.forward_tid = 0
        mock_event.extra_fields.end_time_ns = 2000

        sorted_events = [mock_event]
        with patch("torch_npu.profiler.analysis.prof_parse._event_tree_parser.print_error_msg") as mock_print:
            result = build_event_tree(sorted_events)
            self.assertIsNone(result)
            mock_print.assert_called()

    def test_parse_tensor_metadata_invalid_fields(self):
        tensor_str = "0x12345678;0x87654321;fload32;4;1,2,3;1,2,3;0"
        result = parse_tensor_metadata(tensor_str)
        self.assertIsNone(result)


    def test_parse_input_from_string_none_inputs(self):
        result = parse_input_from_string(None, None, None, None)
        self.assertEqual(result, [])

    def test_build_event_tree_valid_events(self):
        mock_event1 = MagicMock()
        mock_event1.finished = False
        mock_event1.tid = 1
        mock_event1.start_time_ns = 1000
        mock_event1.end_time_ns = 3000
        mock_event1.parent = None
        mock_event1.children = []
        mock_event1.tag = _EventType.TorchOp
        mock_event1.extra_fields = MagicMock()
        mock_event1.extra_fields.forward_tid = 0
        mock_event1.extra_fields.end_time_ns = 2500

        mock_event2 = MagicMock()
        mock_event2.finished = False
        mock_event2.tid = 1
        mock_event2.start_time_ns = 2000
        mock_event2.end_time_ns = 2500
        mock_event2.parent = None
        mock_event2.children = []
        mock_event2.tag = _EventType.TorchOp
        mock_event2.extra_fields = MagicMock()
        mock_event2.extra_fields.forward_tid = 0
        mock_event2.extra_fields.end_time_ns = 2500

        sorted_events = [mock_event1, mock_event2]

        with patch('torch_npu.profiler.analysis.prof_parse._event_tree_parser.print_error_msg') as mock_print:
            result = build_event_tree(sorted_events)
            self.assertIsNone(result)
            mock_print.assert_not_called()

    def test_push_event_children_not_finished(self):
        mock_event = MagicMock()
        mock_event.finished = False
        mock_event.parent = None
        mock_event.children = [MagicMock()]
        mock_event.children[0].finished = False
        mock_event.tid = 1
        mock_event.start_time_ns = 1000
        mock_event.end_time_ns = 2000
        mock_event.tag = _EventType.TorchOp
        mock_event.extra_fields = MagicMock()
        mock_event.extra_fields.forward_tid = 0
        mock_event.extra_fields.end_time_ns = 2000

        thread_event = {}
        unfinished_events = MagicMock()
        with patch('torch_npu.profiler.analysis.prof_parse._event_tree_parser.print_error_msg') as mock_print:
            result = push_event(mock_event, thread_event, unfinished_events)
            self.assertFalse(result)
            mock_print.assert_called_once()

    def test_mark_finished_already_finished(self):
        mock_event = MagicMock()
        mock_event.finished = True
        with patch('torch_npu.profiler.analysis.prof_parse._event_tree_parser.print_error_msg') as mock_print:
            result = mark_finished(mock_event)
            self.assertFalse(result)
            mock_print.assert_called_once()

    def test_push_event_with_parent_already_set(self):
        mock_event = MagicMock()
        mock_event.finished = False
        mock_event.tid = 1
        mock_event.start_time_ns = 1000
        mock_event.end_time_ns = 2000
        mock_event.parent = MagicMock()
        mock_event.children = []
        mock_event.tag = _EventType.TorchOp
        mock_event.extra_fields = MagicMock()
        mock_event.extra_fields.forward_tid = 0
        mock_event.extra_fields.end_time_ns = 2000

        thread_event = {}
        unfinished_events = MagicMock()
        unfinished_events.put = [MagicMock()]

        with patch('torch_npu.profiler.analysis.prof_parse._event_tree_parser.print_error_msg') as mock_print:
            result = push_event(mock_event, thread_event, unfinished_events)
            self.assertFalse(result)
            mock_print.assert_called()


if __name__ == '__main__':
    run_tests()