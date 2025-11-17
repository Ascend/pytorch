import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.utils.affinity import _set_thread_affinity, _reset_thread_affinity


class TestAffinity(TestCase):

    def test_reset_thread_affinity(self):
        original_func = torch_npu._C._npu_reset_thread_affinity
        call_count = 0

        def mock_npu_reset_thread_affinity():
            nonlocal call_count
            call_count += 1

        torch_npu._C._npu_reset_thread_affinity = mock_npu_reset_thread_affinity
        try:
            _reset_thread_affinity()
            self.assertEqual(call_count, 1)
        finally:
            torch_npu._C._npu_reset_thread_affinity = original_func


    def test_set_thread_affinity_invalid_length(self):
        with self.assertRaises(ValueError) as context:
            _set_thread_affinity([1, 2, 3])
        self.assertIn("The length of input list of set_thread_affinity should be 2", str(context.exception))

        with self.assertRaises(ValueError) as context:
            _set_thread_affinity([])
        self.assertIn("The length of input list of set_thread_affinity should be 2", str(context.exception))

    def test_set_thread_affinity_negative_values(self):
        with self.assertRaises(ValueError) as context:
            _set_thread_affinity([-1, 5])
        self.assertIn("Core range should be nonnegative", str(context.exception))

        with self.assertRaises(ValueError) as context:
            _set_thread_affinity([2, -3])
        self.assertIn("Core range should be nonnegative", str(context.exception))

    def test_set_thread_affinity_valid_range(self):
        original_func = torch_npu._C._npu_set_thread_affinity
        call_args = []

        def mock_npu_set_thread_affinity(start, end):
            call_args.append((start, end))

        torch_npu._C._npu_set_thread_affinity = mock_npu_set_thread_affinity
        try:
            _set_thread_affinity([2, 5])
            self.assertEqual(call_args, [(2, 5)])
        finally:
            torch_npu._C._npu_set_thread_affinity = original_func

    def test_set_thread_affinity_none(self):
        original_func = torch_npu._C._npu_set_thread_affinity
        call_args = []

        def mock_npu_set_thread_affinity(start, end):
            call_args.append((start, end))

        torch_npu._C._npu_set_thread_affinity = mock_npu_set_thread_affinity
        try:
            _set_thread_affinity(None)
            self.assertEqual(call_args, [(-1, -1)])
        finally:
            torch_npu._C._npu_set_thread_affinity = original_func

if __name__ == '__main__':
    run_tests()