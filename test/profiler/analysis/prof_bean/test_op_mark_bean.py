import struct
import random

from torch_npu.profiler.analysis.prof_bean._op_mark_bean import OpMarkBean
from torch_npu.profiler.analysis.prof_common_func._constant import Constant
from torch_npu.testing.testcase import TestCase, run_tests


class TestOpMarkBean(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.samples = cls.generate_samples()
        

    @classmethod
    def generate_samples(cls):
        samples = []
        for i in range(4):
            time_ns = random.randint(0, 2**63 - 1)
            corr_id = random.randint(0, 2**64 - 1)
            category = i
            thread_id = random.randint(0, 2**64 - 1)
            process_id = random.randint(0, 2**64 - 1)
            sample = {
                Constant.CONSTANT_BYTES: struct.pack(
                    "<q4Q", time_ns, category, corr_id, thread_id, process_id),
                "time_ns": time_ns, "category": category, "corr_id": corr_id,
                "thread_id": thread_id, "process_id": process_id, "is_enqueue_start": category == 0,
                "is_enqueue_end": category == 1, "is_dequeue_start": category == 2,
                "is_dequeue_end": category == 3
            }
            samples.append(sample)

        return samples

    def test_property(self):
        for sample in self.samples:
            op_mark_bean = OpMarkBean(sample)
            self.assertEqual(sample.get("process_id"), op_mark_bean.pid)
            self.assertEqual(sample.get("thread_id"), op_mark_bean.tid)
            self.assertEqual(sample.get("corr_id"), op_mark_bean.corr_id)
            self.assertEqual(sample.get("is_enqueue_start"), op_mark_bean.is_enqueue_start)
            self.assertEqual(sample.get("is_enqueue_end"), op_mark_bean.is_enqueue_end)
            self.assertEqual(sample.get("is_dequeue_start"), op_mark_bean.is_dequeue_start)
            self.assertEqual(sample.get("is_dequeue_end"), op_mark_bean.is_dequeue_end)
            op_mark_bean.ts = 1099
            self.assertEqual(1099, op_mark_bean.ts)
            op_mark_bean.dur = 876.4
            self.assertEqual(876.4, op_mark_bean.dur)


if __name__ == "__main__":
    run_tests()
