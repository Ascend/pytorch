import time
import datetime
import pickle

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestMemorySnapshotTimeUs(TestCase):
    """Verify that memory snapshot trace events carry time_us (epoch
    microseconds) after enabling _record_memory_history."""

    def setUp(self):
        super().setUp()
        torch_npu.npu.memory._record_memory_history(
            enabled="all", context="all", stacks="python")

    def tearDown(self):
        torch_npu.npu.memory._record_memory_history(enabled=None)

    def _get_entries(self):
        snap = torch_npu.npu.memory._snapshot()
        return [e for tr in snap["device_traces"] for e in tr]

    def test_time_us_present_on_all_events(self):
        """Every trace event in the snapshot carries a time_us field."""
        torch.randn(64, 64, device="npu")
        torch_npu.npu.empty_cache()
        entries = self._get_entries()
        self.assertGreater(len(entries), 0)
        for e in entries:
            self.assertIn("time_us", e)

    def test_time_us_monotonic(self):
        """time_us values are non-decreasing in trace order (TSC is
        monotonic and the converter preserves ordering)."""
        torch.randn(64, 64, device="npu")
        torch_npu.npu.empty_cache()
        entries = self._get_entries()
        times = [e["time_us"] for e in entries if "time_us" in e]
        self.assertGreater(len(times), 0)
        for a, b in zip(times, times[1:]):
            self.assertLessEqual(a, b)

    def test_time_us_epoch_window(self):
        """time_us values fall within a reasonable epoch window (within
        1 hour before now, rejecting raw ticks or bogus values)."""
        torch.randn(64, 64, device="npu")
        torch_npu.npu.empty_cache()
        entries = self._get_entries()
        now = time.time()
        lo = (now - 3600) * 1e6
        hi = (now + 60) * 1e6
        for e in entries:
            t = e["time_us"]
            self.assertGreaterEqual(t, lo)
            self.assertLessEqual(t, hi)

    def test_time_us_readable_datetime(self):
        """time_us converts to a datetime matching the current date."""
        torch.randn(64, 64, device="npu")
        torch_npu.npu.empty_cache()
        entries = self._get_entries()
        times = [e["time_us"] for e in entries if "time_us" in e]
        self.assertGreater(len(times), 0)
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        for t in times:
            dt = datetime.datetime.fromtimestamp(t / 1e6)
            self.assertEqual(dt.strftime("%Y-%m-%d"), today)

    def test_dump_snapshot_pickle_contains_time_us(self):
        """_dump_snapshot pickle round-trips with time_us in every event."""
        import tempfile
        import os
        torch.randn(64, 64, device="npu")
        torch_npu.npu.empty_cache()
        fd, path = tempfile.mkstemp(suffix=".pickle")
        os.close(fd)
        try:
            torch_npu.npu.memory._dump_snapshot(path)
            with open(path, "rb") as f:
                snap = pickle.load(f)
            entries = [e for tr in snap["device_traces"] for e in tr]
            self.assertGreater(len(entries), 0)
            for e in entries:
                self.assertIn("time_us", e)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    run_tests()
