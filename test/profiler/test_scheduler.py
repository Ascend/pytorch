import torch_npu
from torch_npu.profiler import ProfilerAction
from torch_npu.profiler import schedule
from torch_npu.testing.testcase import TestCase, run_tests


class TestScheduler(TestCase):

    def setUp(self):
        self.schedule_list = [
            # wait, active, warmup, repeat, skip_first, [step, expected result]
            [
                0, 0, 0, 0, 0,
                [0, ProfilerAction.RECORD_AND_SAVE],
                [1, ProfilerAction.RECORD_AND_SAVE],
                [10, ProfilerAction.RECORD_AND_SAVE]
            ],
            [
                -1, -1, -1, -1, -1,
                [0, ProfilerAction.RECORD_AND_SAVE],
                [1, ProfilerAction.RECORD_AND_SAVE],
                [10, ProfilerAction.RECORD_AND_SAVE]
            ],
            [
                2, 2, 2, 0, 2,
                [0, ProfilerAction.NONE],
                [1, ProfilerAction.NONE],
                [3, ProfilerAction.NONE],
                [5, ProfilerAction.WARMUP],
                [6, ProfilerAction.RECORD],
                [7, ProfilerAction.RECORD_AND_SAVE],
                [13, ProfilerAction.RECORD_AND_SAVE]
            ],
            [
                2, 2, 2, 1, 2,
                [0, ProfilerAction.NONE],
                [1, ProfilerAction.NONE],
                [3, ProfilerAction.NONE],
                [5, ProfilerAction.WARMUP],
                [6, ProfilerAction.RECORD],
                [7, ProfilerAction.RECORD_AND_SAVE],
                [13, ProfilerAction.NONE]
            ],
            [
                2, 2, 2, 4, 2,
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
                skip_first=samples[4]
            )
            test_pair = samples[self.test_pair_idx:]
            self.assertTrue(sche_inst.active >= 1)
            for step, expect_result in test_pair:
                result = sche_inst(step)
                self.assertEqual(result, expect_result)


if __name__ == "__main__":
    run_tests()