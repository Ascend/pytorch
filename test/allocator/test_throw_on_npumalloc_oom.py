"""Tests for throw_on_npumalloc_oom feature in torch_npu allocator.

This feature preemptively rejects allocations exceeding per_process_memory_fraction
limits, throwing OutOfMemoryError instead of letting the driver attempt a potentially
fatal allocation. Useful for inference serving scenarios where the process must stay alive.
"""

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


# Default settings restored after each test to avoid leaking configuration into
# the rest of the suite (set_allocator_settings is process-global).
_DEFAULT_SETTINGS = "throw_on_npumalloc_oom:False,per_process_memory_fraction:1.0"


class TestThrowOnNpuMallocOOM(TestCase):
    """Test throw_on_npumalloc_oom allocator configuration."""

    def setUp(self):
        # Restore defaults + clear cached blocks so each test starts from a
        # clean slate: prior tests may have left cached blocks or non-default
        # fraction/throw settings behind.
        torch_npu._C._npu_npuCachingAllocator_set_allocator_settings(_DEFAULT_SETTINGS)
        torch_npu.npu.empty_cache()
        torch_npu.npu.reset_accumulated_memory_stats()

    def tearDown(self):
        torch_npu._C._npu_npuCachingAllocator_set_allocator_settings(_DEFAULT_SETTINGS)
        torch_npu.npu.empty_cache()
        torch_npu.npu.reset_accumulated_memory_stats()

    def test_config_parsing_invalid_value(self):
        """Invalid throw_on_npumalloc_oom value raises an error mentioning the
        option name, so callers can diagnose the typo."""
        with self.assertRaisesRegex(RuntimeError, "throw_on_npumalloc_oom"):
            torch_npu._C._npu_npuCachingAllocator_set_allocator_settings(
                "throw_on_npumalloc_oom:maybe"
            )

    def test_oom_rejection_counter_in_stats(self):
        """num_oom_rejections appears in memory_stats with initial value 0
        after reset."""
        stats = torch_npu.npu.memory_stats()
        self.assertIn("num_oom_rejections", stats)
        self.assertEqual(stats["num_oom_rejections"], 0)

    def test_preemptive_rejection_with_low_fraction(self):
        """throw_on_npumalloc_oom:True + a very low fraction preemptively
        rejects an allocation that would exceed the limit with
        OutOfMemoryError, and num_oom_rejections increments."""
        torch_npu._C._npu_npuCachingAllocator_set_allocator_settings(
            "throw_on_npumalloc_oom:True,per_process_memory_fraction:0.01"
        )
        torch_npu.npu.empty_cache()
        torch_npu.npu.reset_accumulated_memory_stats()

        baseline = torch_npu.npu.memory_stats()["num_oom_rejections"]
        # 1GB allocation on a device with only 1% allowed is far above the
        # limit on any realistic NPU (1% of even 16GB = 160MB).
        with self.assertRaises(torch.OutOfMemoryError):
            torch.randn(256, 1024, 1024, device="npu")

        after = torch_npu.npu.memory_stats()["num_oom_rejections"]
        self.assertGreater(after, baseline,
                           "num_oom_rejections should have incremented on rejection")

    def test_normal_alloc_succeeds_without_throw_on_oom(self):
        """throw_on_npumalloc_oom:False bypasses the rejection path entirely
        even when per_process_memory_fraction is restrictive -- the small
        allocation succeeds and num_oom_rejections stays at baseline."""
        torch_npu._C._npu_npuCachingAllocator_set_allocator_settings(
            "throw_on_npumalloc_oom:False,per_process_memory_fraction:0.01"
        )
        baseline = torch_npu.npu.memory_stats()["num_oom_rejections"]
        # Small allocation well under fraction limit, must succeed.
        x = torch.randn(10, 10, device="npu")
        self.assertEqual(x.device.type, "npu")
        # No rejection should have fired.
        self.assertEqual(torch_npu.npu.memory_stats()["num_oom_rejections"], baseline)

    def test_small_alloc_succeeds_with_throw_on_oom(self):
        """throw_on_npumalloc_oom:True does not reject allocations that stay
        within the per_process_memory_fraction limit -- a small allocation
        succeeds and num_oom_rejections stays at baseline. Validates the
        rejection guard is `total_allocated + size > max_allowed`, not an
        unconditional rejection when the flag is on."""
        torch_npu._C._npu_npuCachingAllocator_set_allocator_settings(
            "throw_on_npumalloc_oom:True,per_process_memory_fraction:0.01"
        )
        baseline = torch_npu.npu.memory_stats()["num_oom_rejections"]
        # 10x10 float32 = 400B, far below the fraction limit on any realistic
        # NPU (1% of 16GB = 160MB).
        x = torch.randn(10, 10, device="npu")
        self.assertEqual(x.device.type, "npu")
        # No rejection should have fired for a small allocation.
        self.assertEqual(torch_npu.npu.memory_stats()["num_oom_rejections"], baseline)


if __name__ == "__main__":
    run_tests()
