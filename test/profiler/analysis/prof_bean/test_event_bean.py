from torch_npu.profiler.analysis.prof_bean._event_bean import EventBean
from torch_npu.testing.testcase import TestCase, run_tests


class TestEventBean(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.event_list = [
            {
                "name": "Node@launch",
                "pid": 76820902,
                "tid": 772226,
                "ts": "168.320\t",
                "dur": 14.719999999,
                "args": {
                    "Thread Id": 772226,
                    "Mode": "launch",
                    "level": "node",
                    "id": "0",
                    "item_id": "Sub",
                    "connection_id": 3756,
                },
                "ph": "X",
                # The following is the auxiliary data
                "ts_ns": 168320,
                "end_ns": 183039,
                "is_ai_core": False,
                "unique_id": "76820902-772226-168320"
            },
            {
                "name": "MatMul",
                "pid": 800,
                "tid": 16,
                "ts": "17669.780\t",
                "dur": 160.26,
                "args": {
                    "Model Id": 4294967295,
                    "Task Type": "AI_CORE",
                    "Stream Id": 16,
                    "Task Id": 13489,
                    "Batch Id": 0,
                    "Subtask Id": 4294967295,
                    "connection_id": 3013,
                },
                "ph": "X",
                # The following is the auxiliary data
                "ts_ns": 17669780,
                "end_ns": 17830040,
                "is_ai_core": True,
                "unique_id": "800-16-17669780"
            },
            {
                "name": "thread_name",
                "pid": 800,
                "tid": 0,
                "args": {
                    "name": "Stream 0"
                },
                "ph": "M",
                # The following is the auxiliary data
                "ts_ns": 0,
                "end_ns": 0,
                "is_ai_core": False,
                "unique_id": "800-0-0"
            }
        ]

    def test_property(self):
        for event in self.event_list:
            event_bean = EventBean(event)
            # assert the value of event_bean property
            self.assertEqual(event.get("ts_ns", 0), event_bean.ts)
            self.assertEqual(event.get("pid", ""), event_bean.pid)
            self.assertEqual(event.get("tid", 0), event_bean.tid)
            self.assertEqual(event.get("dur", 0), event_bean.dur)
            self.assertEqual(event.get("end_ns", 0), event_bean.end_ns)
            self.assertEqual(event.get("name", ""), event_bean.name)
            self.assertEqual(event.get("id"), event_bean.id)
            self.assertEqual(event.get("is_ai_core", False), event_bean.is_ai_core)
            self.assertEqual(event.get("unique_id", ""), event_bean.unique_id)


if __name__ == "__main__":
    run_tests()
