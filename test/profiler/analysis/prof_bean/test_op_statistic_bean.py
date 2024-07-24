from torch_npu.profiler.analysis.prof_bean._op_statistic_bean import OpStatisticBean
from torch_npu.testing.testcase import TestCase, run_tests


class TestOpStatisticBean(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.test_cases = [
            {
                "OP Type": "MatMul",
                "Core Type": "AI_CORE",
                "Count": "25",
                "Total Time(us)": "878.9",
                "Min Time(us)": "4.6",
                "Avg Time(us)": "35.156",
                "Max Time(us)": "143.5",
                "Ratio(%)": "2.5"
            },
        ]

    def test_property(self):
        for test_case in self.test_cases:
            op_statistic_bean = OpStatisticBean(test_case)
            self.assertEqual(set(test_case.keys()), set(op_statistic_bean.headers))
            self.assertEqual(set(test_case.values()), set(op_statistic_bean.row))


if __name__ == "__main__":
    run_tests()
