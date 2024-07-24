import random
import struct
from torch_npu.profiler.analysis.prof_bean._gc_record_bean import GCRecordBean
from torch_npu.testing.testcase import TestCase, run_tests


class TestGCRecordBean(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.sample_num = 3
        cls.test_cases = [cls.generate_sample() for _ in range(cls.sample_num)]

    @classmethod
    def generate_sample(cls):
        process_id = random.randint(0, 2**64 - 1)
        start_ns = random.randint(0, 2**64 - 1)
        end_ns = random.randint(0, 2**64 - 1)
        sample = {
            "data": struct.pack("<3Q", process_id, start_ns, end_ns),
            "process_id": process_id, "start_ns": start_ns, "end_ns": end_ns
        }
        return sample

    def test_property(self):
        for test_case in self.test_cases:
            gc_record_bean = GCRecordBean(test_case.get("data"))
            self.assertEqual(test_case.get("process_id"), gc_record_bean.pid)
            self.assertEqual(test_case.get("start_ns"), gc_record_bean.ts)
            self.assertEqual(test_case.get("end_ns") - test_case.get("start_ns"), gc_record_bean.dur)
            self.assertEqual("GC", gc_record_bean.name)


if __name__ == "__main__":
    run_tests()
