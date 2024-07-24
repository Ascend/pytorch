from torch_npu.profiler.analysis.prof_bean._ge_op_memory_bean import GeOpMemoryBean
from torch_npu.testing.testcase import TestCase, run_tests


class TestGeOpMemoryBean(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.test_cases = [
            {
                "Name": "Graph_21",
                "Size(KB)": "192",
                "Allocation Time(us)": "16995.950\t",
                "Duration(us)": "92.6",
                "Allocation Total Allocated(KB)": "192",
                "Allocation Total Reserved(KB)": "502336",
                "Release Total Allocated(KB)": "0",
                "Release Total Reserved(KB)": "502336",
                "Device": "NPU:0",
                # The following is the auxiliary data
                "row": ["cann::Graph_21", "192", "16995.950\t", "17088.550\t", None, "92.6", None,
                        0.1875, 490.5625, None, 0.0, 490.5625, None, None, "NPU:0"],
                "name": "cann::Graph_21",
                "release_time": "17088.550\t",
                "total_alloc": 0.1875,
                "total_reserve": 490.5625,
                "release_total_alloc": 0,
                "release_total_reserve": 490.5625
            },
        ]

    def test_property(self):
        for test_case in self.test_cases:
            ge_op_mem = GeOpMemoryBean(test_case)
            self.assertEqual(test_case.get("row"), ge_op_mem.row)
            self.assertEqual(test_case.get("name"), ge_op_mem.name)
            self.assertEqual(test_case.get("Size(KB)"), ge_op_mem.size)
            self.assertEqual(test_case.get("Allocation Time(us)"), ge_op_mem.allocation_time)
            self.assertEqual(test_case.get("Duration(us)"), ge_op_mem.dur)
            self.assertEqual(test_case.get("release_time"), ge_op_mem.release_time)
            self.assertEqual(test_case.get("total_alloc"), ge_op_mem.allocation_total_allocated)
            self.assertEqual(test_case.get("total_reserve"), ge_op_mem.allocation_total_reserved)
            self.assertEqual(test_case.get("release_total_alloc"), ge_op_mem.release_total_allocated)
            self.assertEqual(test_case.get("release_total_reserve"), ge_op_mem.release_total_reserved)
            self.assertEqual(test_case.get("Device"), ge_op_mem.device)


if __name__ == "__main__":
    run_tests()
