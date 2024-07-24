from collections import OrderedDict

from torch_npu.profiler.analysis.prof_bean._l2_cache_bean import L2CacheBean
from torch_npu.testing.testcase import TestCase, run_tests


class TestL2CacheBean(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        valid_data = OrderedDict()
        valid_data["Stream Id"] = 3
        valid_data["Task Id"] = 13
        valid_data["Hit Rate"] = 0.2
        valid_data["Victim Rate"] = 0.5
        valid_data["Op Name"] = "MatMul"
        cls.test_cases = [
            # data, keys, values, exception
            ["string_data", None, None, AttributeError],
            [[1, 2, 3], None, None, AttributeError],
            [valid_data, ["Stream Id", "Task Id", "Hit Rate", "Victim Rate", "Op Name"],
             [3, 13, 0.2, 0.5, "MatMul"], None],
        ]

    def test_row(self):
        for test_case in self.test_cases:
            data, _, values, exception = test_case
            l2_cahce_bean = L2CacheBean(data)
            if exception:
                with self.assertRaises(exception):
                    _ = l2_cahce_bean.row
                continue
            self.assertEqual(values, l2_cahce_bean.row)

    def test_headers(self):
        for test_case in self.test_cases:
            data, keys, _, exception = test_case
            l2_cache_bean = L2CacheBean(data)
            if exception:
                with self.assertRaises(exception):
                    _ = l2_cache_bean.headers
                continue
            self.assertEqual(keys, l2_cache_bean.headers)


if __name__ == "__main__":
    run_tests()
