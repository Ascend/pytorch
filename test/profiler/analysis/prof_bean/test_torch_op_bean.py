import random
import struct

from torch_npu.profiler.analysis.prof_bean._torch_op_bean import TorchOpBean
from torch_npu.profiler.analysis.prof_common_func._constant import Constant
from torch_npu.testing.testcase import TestCase, run_tests


class TestTorchOpBean(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.sample_num = 3
        cls.test_cases = [cls.generate_sample() for _ in range(cls.sample_num)]

    @classmethod
    def generate_sample(cls):
        start_ns = random.randint(0, 2**63 - 1)
        end_ns = random.randint(0, 2**63 - 1)
        sequence_number = random.randint(0, 2**63 - 1)
        process_id = random.randint(0, 2**64 - 1)
        start_thread_id = random.randint(0, 2**64 - 1)
        end_thread_id = random.randint(0, 2**64 - 1)
        forward_thread_id = random.randint(0, 2**64 - 1)
        scope = random.randint(0, 2**8 - 1)
        is_async = random.choice([0, 1])
        name_id = random.randint(0, 100)

        sample = {
            Constant.CONSTANT_BYTES: struct.pack(
                "<3q4QB?", start_ns, end_ns, sequence_number, process_id,
                start_thread_id, end_thread_id, forward_thread_id, scope, is_async),
            "start_ns": start_ns, "end_ns": end_ns, "sequence_number": sequence_number,
            "process_id": process_id, "start_thread_id": start_thread_id, "end_thread_id": end_thread_id,
            "forward_thread_id": forward_thread_id, "scope": scope, "is_async": is_async, "dur": end_ns - start_ns,
            "args": {Constant.SEQUENCE_NUMBER: sequence_number, Constant.FORWARD_THREAD_ID: forward_thread_id}
        }

        return sample

    def test_property(self):
        for test_case in self.test_cases:
            torch_op_bean = TorchOpBean(test_case)
            self.assertEqual(test_case.get("process_id"), torch_op_bean.pid)
            self.assertEqual(test_case.get("start_thread_id"), torch_op_bean.tid)
            self.assertEqual(test_case.get("scope"), torch_op_bean.scope)
            self.assertEqual(test_case.get("name", ""), torch_op_bean.name)
            self.assertEqual(test_case.get("start_ns"), torch_op_bean.ts)
            self.assertEqual(test_case.get("dur"), torch_op_bean.dur)
            self.assertEqual(test_case.get("end_ns"), torch_op_bean.end_ns)
            self.assertEqual(test_case.get("args"), torch_op_bean.args)
            self.assertEqual(True, torch_op_bean.is_torch_op)


if __name__ == "__main__":
    run_tests()
