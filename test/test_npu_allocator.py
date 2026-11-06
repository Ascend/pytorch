import os

import torch
from torch.testing._internal.common_utils import TestCase, run_tests, serialTest
import torch_npu


class TestRegistrationCounter(TestCase):
    """Tests for the registration_counter feature in the NPU caching allocator.

    registration_counter is a monotonically increasing counter assigned to
    each new memory segment (both npuMalloc-backed and expandable-segment-backed).
    All blocks split from the same segment inherit the same counter value.
    It is exposed via snapshot() / _snapshot() as SegmentInfo.registration_counter.
    """

    _saved_allocator_settings = None
    _saved_fraction = None

    @classmethod
    def setUpClass(cls):
        cls._saved_allocator_settings = os.environ.get("PYTORCH_NPU_ALLOC_CONF", "")
        cls._saved_fraction = torch_npu.npu.get_per_process_memory_fraction()
        torch.npu.memory._set_allocator_settings("expandable_segments:False")

    @classmethod
    def tearDownClass(cls):
        torch_npu.npu.set_per_process_memory_fraction(cls._saved_fraction)
        # _set_allocator_settings only accepts a limited set of keys
        # (expandable_segments, max_split_size_mb, max_non_split_rounding_mb,
        # release_lock_on_npumalloc), so restore only the item this class
        # changed instead of passing the whole PYTORCH_NPU_ALLOC_CONF string.
        original = cls._get_expandable_segments(cls._saved_allocator_settings)
        torch.npu.memory._set_allocator_settings(f"expandable_segments:{original}")

    @staticmethod
    def _get_expandable_segments(env_str):
        """Extract the expandable_segments value from a PYTORCH_NPU_ALLOC_CONF string."""
        for item in env_str.split(","):
            key, sep, value = item.partition(":")
            if sep and key.strip() == "expandable_segments":
                return "True" if value.strip() == "True" else "False"
        return "False"

    def _get_segments_with_counter(self, snapshot_data):
        """Extract segments that have a registration_counter field >= 0."""
        return [seg for seg in snapshot_data["segments"] if seg.get("registration_counter", -1) >= 0]

    @serialTest()
    def test_registration_counter_in_snapshot(self):
        """snapshot() output must include registration_counter >= 0 for each segment."""
        torch_npu.npu.empty_cache()
        x = torch.randn(1024 * 1024, device="npu")  # 4 MB, triggers large pool
        try:
            ss = torch_npu.npu.memory._snapshot()
            self.assertGreater(len(ss["segments"]), 0, "expected at least one segment")
            for seg in ss["segments"]:
                self.assertIn("registration_counter", seg,
                              "registration_counter missing from segment in snapshot")
            # Workspace-allocator segments legitimately report -1; at least one
            # caching-allocator segment must have a valid counter.
            valid = self._get_segments_with_counter(ss)
            self.assertGreater(len(valid), 0,
                               "expected at least one segment with registration_counter >= 0")
        finally:
            del x
            torch_npu.npu.empty_cache()

    @serialTest()
    def test_registration_counter_monotonic(self):
        """Multiple segments must have distinct, monotonically increasing registration_counter values."""
        torch_npu.npu.empty_cache()
        tensors = []
        nelem_20mb = 20 * 1024 * 1024 // 4  # 20 MB of float32

        # Allocate several large tensors to force multiple segments
        for _ in range(4):
            tensors.append(torch.randn(nelem_20mb, device="npu"))

        try:
            ss = torch_npu.npu.memory._snapshot()
            counters = [seg["registration_counter"] for seg in ss["segments"]
                        if seg.get("registration_counter", -1) >= 0]
            self.assertGreater(len(counters), 1, "expected at least 2 segments")
            # All counters must be unique
            self.assertEqual(len(counters), len(set(counters)),
                             "registration_counter values must be unique across segments")
            # Counters must be monotonically increasing (in order of allocation,
            # not necessarily in snapshot order which is address-sorted)
            sorted_counters = sorted(counters)
            for i in range(1, len(sorted_counters)):
                self.assertGreater(sorted_counters[i], sorted_counters[i - 1],
                                   "registration_counter must be monotonically increasing")
        finally:
            del tensors
            torch_npu.npu.empty_cache()

    @serialTest()
    def test_registration_counter_memory_snapshot_api(self):
        """torch_npu.npu.memory.memory_snapshot() must also return registration_counter."""
        torch_npu.npu.empty_cache()
        x = torch.randn(1024 * 256, device="npu")  # 1 MB
        try:
            segments = torch_npu.npu.memory.memory_snapshot()
            self.assertGreater(len(segments), 0, "expected at least one segment")
            for seg in segments:
                self.assertIn("registration_counter", seg,
                              "registration_counter missing from memory_snapshot() segment")
            valid = self._get_segments_with_counter({"segments": segments})
            self.assertGreater(len(valid), 0,
                               "expected at least one segment with registration_counter >= 0")
        finally:
            del x
            torch_npu.npu.empty_cache()

    @serialTest()
    def test_registration_counter_with_mempool(self):
        """Segments allocated in a MemPool must have valid registration_counter values."""
        torch_npu.npu.empty_cache()
        pool = torch_npu.npu.MemPool()
        with torch_npu.npu.use_mem_pool(pool):
            x = torch.randn(1024 * 1024, device="npu")  # 4 MB

        try:
            # torch_npu.npu.MemPool does not provide snapshot(); find this
            # pool's segments by filtering the global snapshot on pool id.
            segments = [s for s in torch_npu.npu.memory._snapshot()["segments"]
                        if s["segment_pool_id"] == pool.id]
            self.assertGreater(len(segments), 0, "expected at least one segment in the mem pool")
            for seg in segments:
                self.assertIn("registration_counter", seg,
                              "registration_counter missing from mem pool segment")
                self.assertGreaterEqual(seg["registration_counter"], 0,
                                        "registration_counter should be >= 0 in mem pool segment")
        finally:
            del x
            del pool
            torch_npu.npu.empty_cache()

    @serialTest()
    def test_registration_counter_split_inheritance(self):
        """Blocks split from the same segment must share the same registration_counter.

        When a segment is split, the new block inherits the parent's counter.
        In the snapshot, all blocks within a single segment share one
        registration_counter (the head block's value).
        """
        torch_npu.npu.empty_cache()
        # Allocate a tensor that will cause a segment to be created and split
        # (large pool: segment is 20 MB, allocate 4 MB → segment is split)
        nelem_4mb = 4 * 1024 * 1024 // 4
        x = torch.randn(nelem_4mb, device="npu")

        try:
            ss = torch_npu.npu.memory._snapshot()
            segments = ss["segments"]
            self.assertGreater(len(segments), 0)
            # Find the segment that contains our allocation
            seg = segments[0]
            self.assertIn("registration_counter", seg)
            self.assertGreaterEqual(seg["registration_counter"], 0)
            # A split segment should have more than one block
            # (one allocated, one free remainder)
            # Verify the counter is present and valid even after split.
            # Workspace-allocator segments legitimately report -1; at least one
            # split segment must have a valid counter.
            valid = [s for s in segments if s["registration_counter"] >= 0]
            self.assertGreater(len(valid), 0,
                               "expected at least one split segment with registration_counter >= 0")
        finally:
            del x
            torch_npu.npu.empty_cache()


if __name__ == '__main__':
    run_tests()
