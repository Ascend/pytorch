from torch_npu.profiler.analysis.prof_bean._op_summary_bean import OpSummaryBean
from torch_npu.profiler.analysis.prof_common_func._csv_headers import CsvHeaders
from torch_npu.testing.testcase import TestCase, run_tests


class TestOpSummaryBean(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.test_cases = [
            [
                {
                    "Model ID": "899",
                    "Task ID": "9469",
                    "Op Name": "MatMul",
                    "Task Type": "AI_CORE",
                    "Task Start Time": "149999",
                    "Task Duratioin(us)": "121.88",
                    "Task Wait Time(us)": "0.11",
                },
                {
                    "headers": [],
                    "expect_row": []
                }
            ],
            [
                {
                    "Model ID": "899",
                    "Task ID": "9469",
                    "Op Name": "MatMul",
                    "Task Type": "AI_CORE",
                    "Task Start Time": "149999",
                    "Task Duratioin(us)": "121.88",
                    "Task Wait Time(us)": "0.11",
                },
                {
                    "headers": ["Model ID", "Task ID", "Op Name"],
                    "expect_row": ["899", "9469", "MatMul"]
                }
            ]
        ]

    def test_property(self):
        for data, auxiliary_info in self.test_cases:
            op_summary_bean = OpSummaryBean(data)
            OpSummaryBean.headers = auxiliary_info.get("headers", [])
            if auxiliary_info.get("headers", []):
                self.assertEqual(auxiliary_info.get("expect_row"), op_summary_bean.row)
            else:
                self.assertEqual(set(data.values()), set(op_summary_bean.row))
            self.assertEqual(data.get(CsvHeaders.TASK_START_TIME, "0"), op_summary_bean.ts)
            self.assertEqual(data.keys(), op_summary_bean.all_headers)


if __name__ == "__main__":
    run_tests()
