from unittest.mock import patch, Mock, MagicMock
from torch_npu.profiler.analysis.prof_common_func.constant import Constant
from torch_npu.profiler.analysis.prof_common_func.trace_event_manager import TraceEventManager
from torch_npu.testing.testcase import TestCase, run_tests


class TestTraceEventManager(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def test_x_event(self):
        event = MagicMock()
        event.pid = 999
        event.name = "MatMul"
        event.args = {Constant.INPUT_SHAPES: "[2, 2048]", Constant.CALL_STACK: "call stack string1"}
        event.ts = 2000
        event.end_ns = 30
        event.dur = 1000
        expect = {"ph": "X", "name": event.name, "pid": event.pid, "tid": event.tid, "ts": "2.000",
                "dur": 1.0, "cat": "cpu_op", "args": event.args}
        self.assertEqual(expect, TraceEventManager.create_x_event(event, "cpu_op"))

    def test_create_m_event(self):
        pid = 100
        tid_dict = {200: True}
        expect = [
            {"ph": "M", "name": Constant.PROCESS_NAME, "pid": pid, "tid": 0, "args": {"name": "Python"}},
            {"ph": "M", "name": Constant.PROCESS_LABEL, "pid": pid, "tid": 0, "args": {"labels": "CPU"}},
            {"ph": "M", "name": Constant.PROCESS_SORT, "pid": pid, "tid": 0, "args": {"sort_index": 0}},
            {"ph": "M", "name": Constant.THREAD_NAME, "pid": pid, "tid": 200, "args": {"name": f"Thread {200}"}},
            {"ph": "M", "name": Constant.THREAD_SORT, "pid": pid, "tid": 200, "args": {"sort_index": 201}}
        ]
        self.assertEqual(expect, TraceEventManager.create_m_event(pid, tid_dict))

    def test_create_torch_to_npu_flow(self):
        # start event
        start_event = MagicMock()
        start_event.pid = 999
        start_event.tid = 111
        start_event.name = "aten:matmul"
        start_event.ts = 3000
        # end event
        end_event = MagicMock()
        end_event.pid = 999
        end_event.tid = 222
        end_event.name = "MatMul"
        end_event.ts = 2000
        expect = [
            {"ph": "s", "bp": "e", "name": "torch_to_npu", "id": 2000, "pid": start_event.pid,
             "tid": start_event.tid, "ts": "3.000", "cat": "async_npu"},
            {"ph": "f", "bp": "e", "name": "torch_to_npu", "id": 2000, "pid": end_event.pid,
             "tid": end_event.tid, "ts": "2.000", "cat": "async_npu"}
        ]
        self.assertEqual(expect, TraceEventManager.create_torch_to_npu_flow(start_event, end_event))

    def test_create_task_queue_flow(self):
        que_event = MagicMock()
        que_event.pid = 999
        que_event.tid = 222
        que_event.name = "Enque_OR_Deque"
        que_event.ts = 2000
        expect1 = {"ph": "s", "bp": "e", "name": "enqueue_to_dequeue", "id": que_event.corr_id, "pid": que_event.pid,
                   "tid": que_event.tid, "ts": "2.000", "cat": "async_task_queue"}
        expect2 = {"ph": "f", "bp": "e", "name": "enqueue_to_dequeue", "id": que_event.corr_id, "pid": que_event.pid,
                   "tid": que_event.tid, "ts": "2.000", "cat": "async_task_queue"}
        self.assertEqual(expect1, TraceEventManager.create_task_queue_flow("s", que_event))
        self.assertEqual(expect2, TraceEventManager.create_task_queue_flow("f", que_event))

    def test_create_fwd_flow(self):
        events = {
            999: {
                "start": {
                    "pid": 111, "tid": 222, "ts": 3000
                },
                "end": {
                    "pid": 333, "tid": 444, "ts": 4000
                }
            }
        }
        expect = [
            {"ph": "s", "bp": "e", "name": "fwdbwd", "id": 999, "pid": 111,
            "tid": 222, "ts": "3.000", "cat": "fwdbwd"},
            {"ph": "f", "bp": "e", "name": "fwdbwd", "id": 999, "pid": 333,
             "tid": 444, "ts": "4.000", "cat": "fwdbwd"}
        ]
        self.assertEqual(expect, TraceEventManager.create_fwd_flow(events))


if __name__ == "__main__":
    run_tests()
