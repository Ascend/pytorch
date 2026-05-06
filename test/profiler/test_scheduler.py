# Owner(s): ["oncall: profiler"]
from torch_npu.profiler import ProfilerAction, schedule
from torch_npu.testing.testcase import run_tests, TestCase


class TestScheduler(TestCase):
    def setUp(self):
        self.schedule_list = [
            # wait, active, warmup, repeat, skip_first, [step, expected result]
            [
                0,
                0,
                0,
                0,
                0,
                [0, ProfilerAction.RECORD_AND_SAVE],
                [1, ProfilerAction.RECORD_AND_SAVE],
                [10, ProfilerAction.RECORD_AND_SAVE],
            ],
            [
                -1,
                -1,
                -1,
                -1,
                -1,
                [0, ProfilerAction.RECORD_AND_SAVE],
                [1, ProfilerAction.RECORD_AND_SAVE],
                [10, ProfilerAction.RECORD_AND_SAVE],
            ],
            [
                2,
                2,
                2,
                0,
                2,
                [0, ProfilerAction.NONE],
                [1, ProfilerAction.NONE],
                [3, ProfilerAction.NONE],
                [5, ProfilerAction.WARMUP],
                [6, ProfilerAction.RECORD],
                [7, ProfilerAction.RECORD_AND_SAVE],
                [13, ProfilerAction.RECORD_AND_SAVE],
            ],
            [
                2,
                2,
                2,
                1,
                2,
                [0, ProfilerAction.NONE],
                [1, ProfilerAction.NONE],
                [3, ProfilerAction.NONE],
                [5, ProfilerAction.WARMUP],
                [6, ProfilerAction.RECORD],
                [7, ProfilerAction.RECORD_AND_SAVE],
                [13, ProfilerAction.NONE],
            ],
            [
                2,
                2,
                2,
                4,
                2,
                [0, ProfilerAction.NONE],
                [1, ProfilerAction.NONE],
                [3, ProfilerAction.NONE],
                [5, ProfilerAction.WARMUP],
                [6, ProfilerAction.RECORD],
                [7, ProfilerAction.RECORD_AND_SAVE],
                [13, ProfilerAction.RECORD_AND_SAVE],
                [25, ProfilerAction.RECORD_AND_SAVE],
                [30, ProfilerAction.NONE],
            ],
        ]
        self.test_pair_idx = 5

    def test_call(self):
        for samples in self.schedule_list:
            sche_inst = schedule(
                wait=samples[0],
                active=samples[1],
                warmup=samples[2],
                repeat=samples[3],
                skip_first=samples[4],
            )
            test_pair = samples[self.test_pair_idx :]
            self.assertTrue(sche_inst.active >= 1)
            for step, expect_result in test_pair:
                result = sche_inst(step)
                self.assertEqual(result, expect_result)

    def test_skip_first_wait_should_works_when_non_zero(self):
        test_schedule = schedule(
            skip_first=1, wait=2, warmup=1, active=2, repeat=2, skip_first_wait=1
        )
        test_schedule_expected_outputs = [
            # repeat No. 1 begin
            # skip first 1
            ProfilerAction.NONE,
            # warmup 1
            ProfilerAction.WARMUP,
            # active 1 begin
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
            # active 1 end
            # repeat No. 1 end
            # ---
            # repeat No. 2 begin
            # wait 2
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            # warmup 1
            ProfilerAction.WARMUP,
            # active 2 begin
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
            # active 2 end
            # repeat No. 2 end
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
        ]
        for step in range(len(test_schedule_expected_outputs)):
            self.assertEqual(test_schedule(step), test_schedule_expected_outputs[step])

    def test_skip_first_wait_should_be_reset_when_invalid(self):
        test_schedule = schedule(
            skip_first=1, wait=2, warmup=1, active=2, repeat=2, skip_first_wait=0.5
        )
        test_schedule_expected_outputs = [
            # skip first 1
            ProfilerAction.NONE,
            # repeat No. 1 begin
            # wait 2
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            # warmup 1
            ProfilerAction.WARMUP,
            # active 1 begin
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
            # active 1 end
            # repeat No. 1 end
            # ---
            # repeat No. 2 begin
            # wait 2
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            # warmup 1
            ProfilerAction.WARMUP,
            # active 2 begin
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
            # active 2 end
            # repeat No. 2 end
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
        ]
        for step in range(len(test_schedule_expected_outputs)):
            self.assertEqual(test_schedule(step), test_schedule_expected_outputs[step])


if __name__ == "__main__":
    run_tests()
