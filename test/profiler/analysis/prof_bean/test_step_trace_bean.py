from torch_npu.profiler.analysis.prof_bean._step_trace_bean import StepTraceBean
from torch_npu.testing.testcase import TestCase, run_tests


class TestStepTraceBean(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.test_cases = [
            {
                "Iteration ID": 10,
                "expect_step_id": 10
            },
            {
                "expect_step_id": -1
            }
        ]

    def test_property(self):
        for test_case in self.test_cases:
            step_trace_bean = StepTraceBean(test_case)
            self.assertEqual(test_case.get("expect_step_id"), step_trace_bean.step_id)


if __name__ == "__main__":
    run_tests()
