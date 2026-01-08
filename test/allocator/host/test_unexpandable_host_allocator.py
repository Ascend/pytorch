import gc

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestHostCachingAllocator(TestCase):
    def test_host_memory_stats(self):
        # Helper functions
        def empty_stats():
            return {
                "allocated_bytes.allocated": 0,
                "allocated_bytes.current": 0,
                "allocated_bytes.freed": 0,
                "allocated_bytes.peak": 0,
                "allocations.allocated": 0,
                "allocations.current": 0,
                "allocations.freed": 0,
                "allocations.peak": 0,
                "host_alloc_time.count": 0,
                "host_free_time.count": 0,
                "num_host_alloc": 0,
                "num_host_free": 0,
                "active_bytes.allocated": 0,
                "active_bytes.current": 0,
                "active_bytes.freed": 0,
                "active_bytes.peak": 0,
                "active_requests.allocated": 0,
                "active_requests.current": 0,
                "active_requests.freed": 0,
                "active_requests.peak": 0,
            }

        def check_stats(expected):
            stats = torch_npu.npu.host_memory_stats()
            for k, v in expected.items():
                if v != stats[k]:
                    print(f"key: {k}, expected: {v}, stats: {stats[k]}")
                self.assertEqual(v, stats[k])

        # Setup the test cleanly
        alloc1 = 10
        alloc1_aligned = 16
        alloc2 = 20
        alloc2_aligned = 32
        expected = empty_stats()

        # Reset any lingering state
        gc.collect()
        torch_npu.npu.host_empty_cache()

        # Check that stats are empty
        check_stats(expected)

        # Make first allocation and check stats
        t1 = torch.ones(alloc1 * 1024, pin_memory=True)
        self.assertTrue(t1.is_pinned())
        for prefix in ["active_requests", "allocations"]:
            for suffix in ["allocated", "current", "peak"]:
                expected[prefix + "." + suffix] += 1

        allocation_size1 = alloc1_aligned * 1024 * 4
        for prefix in ["allocated_bytes", "active_bytes"]:
            for suffix in ["allocated", "current", "peak"]:
                expected[prefix + "." + suffix] += allocation_size1

        expected["num_host_alloc"] += 1
        expected["host_alloc_time.count"] += 1

        check_stats(expected)

        # Make second allocation and check stats
        t2 = torch.ones(alloc2 * 1024, pin_memory=True)
        self.assertTrue(t2.is_pinned())
        for prefix in ["active_requests", "allocations"]:
            for suffix in ["allocated", "current", "peak"]:
                expected[prefix + "." + suffix] += 1

        allocation_size2 = alloc2_aligned * 1024 * 4
        for prefix in ["allocated_bytes", "active_bytes"]:
            for suffix in ["allocated", "current", "peak"]:
                expected[prefix + "." + suffix] += allocation_size2

        expected["num_host_alloc"] += 1
        expected["host_alloc_time.count"] += 1

        check_stats(expected)

        # Empty cache and check stats
        torch_npu.npu.host_empty_cache()

        check_stats(expected)

        # Finally, check the reset of peak and accumulated stats
        torch_npu.npu.reset_peak_host_memory_stats()
        torch_npu.npu.reset_accumulated_host_memory_stats()


if __name__ == '__main__':
    run_tests()