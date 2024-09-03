import struct
import random

from torch_npu.profiler.analysis.prof_bean._memory_use_bean import MemoryUseBean
from torch_npu.profiler.analysis.prof_common_func._constant import Constant
from torch_npu.testing.testcase import TestCase, run_tests


class TestMemoryUsageBean(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.sample_nums = 3
        cls.npu_id = 20
        cls.test_cases = [cls.generate_sample() for _ in range(cls.sample_nums)]

    @classmethod
    def generate_sample(cls):
        ptr = random.randint(0, 2**63 - 1)
        time_ns = random.randint(0, 2**63 - 1)
        alloc_size = random.randint(0, 2**63 - 1)
        total_alloc = random.randint(0, 2**63 - 1)
        total_reserve = random.randint(0, 2**63 - 1)
        total_active = random.randint(0, 2**63 - 1)
        device_type = random.choice([0, 20])
        data_type = random.choice([0, 9])
        allocator_type = random.choice([0, 9])
        device_index = random.randint(-2**7, 2**7 - 1)
        stream_ptr = random.randint(0, 2**63 - 1)
        thread_id = random.randint(0, 2**64 - 1)
        process_id = random.randint(0, 2**64 - 1)
        sample = {
            Constant.CONSTANT_BYTES: struct.pack(
                "<7q2b2B2Q", ptr, time_ns, alloc_size, total_alloc, total_reserve, total_active, stream_ptr, device_type,
                device_index, data_type, allocator_type, thread_id, process_id),
            "ptr": ptr, "time_ns": time_ns, "alloc_size": alloc_size / Constant.B_TO_KB, "alloc_size_for_db": alloc_size,
            "total_alloc": total_alloc / Constant.B_TO_MB, "total_alloc_for_db": total_alloc,
            "total_reserve": total_reserve / Constant.B_TO_MB, "total_reserve_for_db": total_reserve,
            "total_active": total_active / Constant.B_TO_MB, "total_active_for_db": total_active,
            "stream_ptr": stream_ptr, "dev_type": device_type, "dev_id": device_index,
            "data_type": data_type, "allocator_type": allocator_type, "thread_id": thread_id, "process_id": process_id
        }
        sample["is_npu"] = True if device_type == cls.npu_id else False

        return sample

    def test_property(self):
        for sample in self.test_cases:
            memory_usage_bean = MemoryUseBean(sample)
            self.assertEqual(sample.get("ptr"), memory_usage_bean.ptr)
            self.assertEqual(sample.get("alloc_size"), memory_usage_bean.alloc_size)
            self.assertEqual(sample.get("alloc_size_for_db"), memory_usage_bean.alloc_size_for_db)
            self.assertEqual(sample.get("total_alloc"), memory_usage_bean.total_allocated)
            self.assertEqual(sample.get("total_alloc_for_db"), memory_usage_bean.total_allocated_for_db)
            self.assertEqual(sample.get("total_reserve"), memory_usage_bean.total_reserved)
            self.assertEqual(sample.get("total_reserve_for_db"), memory_usage_bean.total_reserved_for_db)
            self.assertEqual(sample.get("dev_type"), memory_usage_bean.device_type)
            self.assertEqual(sample.get("dev_id"), memory_usage_bean.device_index)
            self.assertEqual(sample.get("thread_id"), memory_usage_bean.tid)
            self.assertEqual(sample.get("process_id"), memory_usage_bean.pid)
            self.assertEqual(sample.get("is_npu"), memory_usage_bean.is_npu())
            self.assertEqual(sample.get("total_active"), memory_usage_bean.total_active)
            self.assertEqual(sample.get("total_active_for_db"), memory_usage_bean.total_active_for_db)
            self.assertEqual(sample.get("data_type"), memory_usage_bean.data_type)
            self.assertEqual(sample.get("allocator_type"), memory_usage_bean.allocator_type)
            self.assertEqual(sample.get("stream_ptr"), memory_usage_bean.stream_ptr)


if __name__ == "__main__":
    run_tests()
