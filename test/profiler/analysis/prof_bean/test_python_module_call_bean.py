import random
import struct
from torch_npu.profiler.analysis.prof_bean.python_module_call_bean import PythonModuleCallBean
from torch_npu.profiler.analysis.prof_common_func.constant import Constant
from torch_npu.testing._testcase import TestCase, run_tests


class TestPythonModuleCallBean(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.sample_num = 3
        cls.test_cases = [cls.generate_sample() for _ in range(cls.sample_num)]

    @classmethod
    def generate_sample(cls):
        idx = random.randint(0, 2**64 - 1)
        thread_id = random.randint(0, 2**64 - 1)
        process_id = random.randint(0, 2**64 - 1)
        sample = {
            Constant.CONSTANT_BYTES: struct.pack("<3Q", idx, thread_id, process_id),
            "idx": idx, "thread_id": thread_id, "process_id": process_id
        }
        return sample

    def test_property(self):
        for test_case in self.test_cases:
            python_mudole_call_bean = PythonModuleCallBean(test_case)
            self.assertEqual(test_case.get("idx"), python_mudole_call_bean.idx)
            self.assertEqual(test_case.get("thread_id"), python_mudole_call_bean.tid)
            self.assertEqual(test_case.get("process_id"), python_mudole_call_bean.pid)
            self.assertEqual(test_case.get("module_uid", ""), python_mudole_call_bean.module_uid)
            self.assertEqual(test_case.get("module_name", ""), python_mudole_call_bean.module_name)


if __name__ == "__main__":
    run_tests()
