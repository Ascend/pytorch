from unittest.mock import MagicMock

from torch_npu.profiler._profiler_action_controller import ProfActionController
from torch_npu.profiler.profiler_interface import _ProfInterface
from torch_npu.profiler import ProfilerAction
from torch_npu.testing.testcase import TestCase, run_tests


class TestActionController(TestCase):

    def setUp(self):
        self.prof_if = _ProfInterface()
        self.prof_if.init_trace = MagicMock(name='init_trace')
        self.prof_if.start_trace = MagicMock(name='start_trace')
        self.prof_if.stop_trace = MagicMock(name='stop_trace')
        self.prof_if.finalize_trace = MagicMock(name='finalize_trace')
        self.on_trace_ready = MagicMock(name='on_trace_ready')
        self.prof_if.delete_prof_dir = MagicMock(name='delete_prof_dir')
        self.action_controller = ProfActionController(self, self.prof_if, self.on_trace_ready)
        self.test_cases = [
            # call_cnt_list: [init_trace, start_trace, stop_trace, finalize_trace, on_trace_ready, delete_prof_dir]
            [ProfilerAction.NONE, ProfilerAction.NONE, [0, 0, 0, 0, 0, 0]],
            [ProfilerAction.NONE, ProfilerAction.WARMUP, [1, 0, 0, 0, 0, 0]],
            [ProfilerAction.NONE, ProfilerAction.RECORD, [2, 1, 0, 0, 0, 0]],
            [ProfilerAction.NONE, ProfilerAction.RECORD_AND_SAVE, [3, 2, 0, 0, 0, 0]],
            [ProfilerAction.WARMUP, ProfilerAction.NONE, [3, 3, 1, 1, 0, 0]],
            [ProfilerAction.WARMUP, ProfilerAction.WARMUP, [3, 3, 1, 1, 0, 0]],
            [ProfilerAction.WARMUP, ProfilerAction.RECORD, [3, 4, 1, 1, 0, 0]],
            [ProfilerAction.WARMUP, ProfilerAction.RECORD_AND_SAVE, [3, 5, 1, 1, 0, 0]],
            [ProfilerAction.RECORD, ProfilerAction.NONE, [3, 5, 2, 2, 0, 0]],
            [ProfilerAction.RECORD, ProfilerAction.WARMUP, [3, 5, 3, 3, 0, 0]],
            [ProfilerAction.RECORD, ProfilerAction.RECORD, [3, 5, 3, 3, 0, 0]],
            [ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE, [3, 5, 3, 3, 0, 0]],
            [ProfilerAction.RECORD_AND_SAVE, ProfilerAction.NONE, [3, 5, 4, 4, 1, 0]],
            [ProfilerAction.RECORD_AND_SAVE, ProfilerAction.WARMUP, [4, 5, 5, 5, 2, 0]],
            [ProfilerAction.RECORD_AND_SAVE, ProfilerAction.RECORD, [5, 6, 6, 6, 3, 0]],
            [ProfilerAction.RECORD_AND_SAVE, ProfilerAction.RECORD_AND_SAVE, [6, 7, 7, 7, 4, 0]],
            [ProfilerAction.WARMUP, None, [6, 7, 7, 8, 4, 1]],
            [ProfilerAction.RECORD, None, [6, 7, 8, 9, 5, 1]],
            [ProfilerAction.RECORD_AND_SAVE, None, [6, 7, 9, 10, 6, 1]],
            [None, None, [6, 7, 9, 10, 6, 1]],
        ]

    def test_transit_action(self):
        for prev, current, call_cnt_list in self.test_cases:
            self.action_controller.transit_action(prev, current)
            self.assertEqual(self.prof_if.init_trace.call_count, call_cnt_list[0])
            self.assertEqual(self.prof_if.start_trace.call_count, call_cnt_list[1])
            self.assertEqual(self.prof_if.stop_trace.call_count, call_cnt_list[2])
            self.assertEqual(self.prof_if.finalize_trace.call_count, call_cnt_list[3])
            self.assertEqual(self.on_trace_ready.call_count, call_cnt_list[4])
            self.assertEqual(self.prof_if.delete_prof_dir.call_count, call_cnt_list[5])


if __name__ == "__main__":
    run_tests()
