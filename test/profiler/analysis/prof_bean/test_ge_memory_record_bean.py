from torch_npu.profiler.analysis.prof_bean._ge_memory_record_bean import GeMemoryRecordBean
from torch_npu.testing.testcase import TestCase, run_tests


class TestGeMemoryRecordBean(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.test_cases = [
            {
                "Component": "GE",
                "Timestamp(us)": "1699.550\t",
                "Total Allocated(KB)": "192",
                "Total Reserved(KB)": "502336",
                "Device": "NPU:0",
                # The following is the auxiliary data
                "row": ["GE", "1699.550\t", 0.1875, 490.5625, None, None, "NPU:0"],
                "time_ns": 1699550,
                "total_alloc": 0.1875,
                "total_reserve": 490.5625
            },
        ]

    def test_property(self):
        for test_case in self.test_cases:
            ge_mem_record = GeMemoryRecordBean(test_case)
            self.assertEqual(test_case.get("row"), ge_mem_record.row)
            self.assertEqual(test_case.get("Component"), ge_mem_record.component)
            self.assertEqual(test_case.get("time_ns"), ge_mem_record.time_ns)
            self.assertEqual(test_case.get("total_alloc"), ge_mem_record.total_allocated)
            self.assertEqual(test_case.get("total_reserve"), ge_mem_record.total_reserved)
            self.assertEqual(test_case.get("Device"), ge_mem_record.device_tag)


if __name__ == "__main__":
    run_tests()
