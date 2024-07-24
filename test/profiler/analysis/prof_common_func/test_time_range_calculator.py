from torch_npu.profiler.analysis.prof_common_func._time_range_calculator import CommunicationTimeRange, RangeCaculator

from torch_npu.testing.testcase import TestCase, run_tests


class TestTimeRangeCalculator(TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def test_generate_time_range(self):
        start_ts = 5
        end_ts = 10
        time_range = RangeCaculator.generate_time_range(start_ts, end_ts)
        self.assertEqual(time_range.start_ts, start_ts)
        self.assertEqual(time_range.end_ts, end_ts)

    def test_merge_continuous_intervals(self):
        time_range_list = [
            RangeCaculator.generate_time_range(12, 45),
            RangeCaculator.generate_time_range(50, 60),
            RangeCaculator.generate_time_range(6, 20)
        ]
        result_list = RangeCaculator.merge_continuous_intervals(time_range_list)
        self.assertEqual(len(result_list), 2)
        self.assertEqual(result_list[0].start_ts, 6)
        self.assertEqual(result_list[0].end_ts, 45)
        self.assertEqual(result_list[1].start_ts, 50)
        self.assertEqual(result_list[1].end_ts, 60)

    def test_compute_pipeline_overlap(self):
        communication_data = [
            RangeCaculator.generate_time_range(30, 80, CommunicationTimeRange),
            RangeCaculator.generate_time_range(90, 120, CommunicationTimeRange),
            RangeCaculator.generate_time_range(130, 150, CommunicationTimeRange),
        ]
        compute_data = [
            RangeCaculator.generate_time_range(6, 45),
            RangeCaculator.generate_time_range(70, 100)
        ]
        pure_communication_data, free_data = \
            RangeCaculator.compute_pipeline_overlap(communication_data, compute_data)
        self.assertEqual(len(pure_communication_data), 3)
        self.assertEqual(pure_communication_data[0].start_ts, 45)
        self.assertEqual(pure_communication_data[0].end_ts, 70)
        self.assertEqual(pure_communication_data[1].start_ts, 100)
        self.assertEqual(pure_communication_data[1].end_ts, 120)
        self.assertEqual(pure_communication_data[2].start_ts, 130)
        self.assertEqual(pure_communication_data[2].end_ts, 150)
        self.assertEqual(len(free_data), 1)
        self.assertEqual(free_data[0].start_ts, 120)
        self.assertEqual(free_data[0].end_ts, 130)


if __name__ == "__main__":
    run_tests()
