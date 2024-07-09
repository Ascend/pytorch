import random
import struct
from torch_npu.profiler.analysis.prof_bean.python_func_call_bean import PythonFuncCallBean
from torch_npu.profiler.analysis.prof_common_func.constant import Constant
from torch_npu.testing.testcase import TestCase, run_tests


class TestPythonFuncCallBean(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.sample_num = 3
        cls.test_cases = [cls.generate_sample() for _ in range(cls.sample_num)]

    @classmethod
    def generate_sample(cls):
        start_ns = random.randint(0, 2**64 - 1)
        thread_id = random.randint(0, 2**64 - 1)
        process_id = random.randint(0, 2**64 - 1)
        trace_tag = random.randint(0, 3)
        sample = {
            Constant.CONSTANT_BYTES: struct.pack("<3QB", start_ns, thread_id, process_id, trace_tag),
            "start_ns": start_ns, "thread_id": thread_id, "process_id": process_id, "trace_tag": trace_tag
        }
        return sample

    def test_property(self):
        for test_case in self.test_cases:
            python_func_call_bean = PythonFuncCallBean(test_case)
            self.assertEqual(test_case.get("start_ns"), python_func_call_bean.start_ns)
            self.assertEqual(test_case.get("thread_id"), python_func_call_bean.tid)
            self.assertEqual(test_case.get("process_id"), python_func_call_bean.pid)
            self.assertEqual(test_case.get("trace_tag"), python_func_call_bean.trace_tag)
            self.assertEqual(test_case.get("name", ""), python_func_call_bean.name)


if __name__ == "__main__":
    run_tests()
