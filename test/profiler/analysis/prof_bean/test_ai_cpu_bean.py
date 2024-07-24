from collections import OrderedDict

from torch_npu.profiler.analysis.prof_bean._ai_cpu_bean import AiCpuBean
from torch_npu.testing.testcase import TestCase, run_tests


class TestAiCPUBean(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        valid_data = OrderedDict()
        valid_data["Timestamps(us)"] = 1768
        valid_data["Node"] = "IndexPutV2"
        valid_data["Compute_time(us)"] = 0.2
        cls.test_cases = [
            # data, keys, values, exception
            ["string_data", None, None, AttributeError],
            [[1, 2, 3], None, None, AttributeError],
            [valid_data, ["Timestamps(us)", "Node", "Compute_time(us)"],
             [1768, "IndexPutV2", 0.2], None],
        ]

    def test_row(self):
        for test_case in self.test_cases:
            data, _, values, exception = test_case
            _ai_cpu_bean = AiCpuBean(data)
            if exception:
                with self.assertRaises(exception):
                    _ = _ai_cpu_bean.row
                continue
            self.assertEqual(values, _ai_cpu_bean.row)

    def test_headers(self):
        for test_case in self.test_cases:
            data, keys, _, exception = test_case
            _ai_cpu_bean = AiCpuBean(data)
            if exception:
                with self.assertRaises(exception):
                    _ = _ai_cpu_bean.headers
                continue
            self.assertEqual(keys, _ai_cpu_bean.headers)


if __name__ == "__main__":
    run_tests()
