from torch_npu.profiler.analysis.prof_bean._api_statistic_bean import ApiStatisticBean
from torch_npu.testing.testcase import TestCase, run_tests


class TestApiStatisticBean(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.test_cases = [
            {
                "Level": "acl",
                "API Name": "aclnnMm",
                "Time(us)": "37.77",
                "Count": "1",
                "Avg(us)": "37.77",
                "Min(us)": "37.77",
                "Max(us)": "37.77",
                "Variance": "0"
            },
        ]

    def test_property(self):
        for test_case in self.test_cases:
            api_statistic_bean = ApiStatisticBean(test_case)
            self.assertEqual(set(test_case.keys()), set(api_statistic_bean.headers))
            self.assertEqual(set(test_case.values()), set(api_statistic_bean.row))


if __name__ == "__main__":
    run_tests()
