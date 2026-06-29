"""Tests for throw_on_npumalloc_oom feature in torch_npu allocator.

This feature preemptively rejects allocations exceeding per_process_memory_fraction
limits, throwing OutOfMemoryError instead of letting the driver attempt a potentially
fatal allocation. Useful for inference serving scenarios where the process must stay alive.
"""

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestThrowOnNpuMallocOOM(TestCase):
    """Test throw_on_npumalloc_oom allocator configuration."""

    def test_config_parsing_valid_true(self):
        """Test that throw_on_npumalloc_oom:True is parsed correctly."""
        torch_npu._C._npu_npuCachingAllocator_set_allocator_settings(
            "throw_on_npumalloc_oom:True,per_process_memory_fraction:0.5"
        )
        # Verify the setting took effect via memory stats
        stats = torch_npu.npu.memory_stats()
        self.assertIn("num_oom_rejections", stats)
        # Reset
        torch_npu._C._npu_npuCachingAllocator_set_allocator_settings(
            "throw_on_npumalloc_oom:False,per_process_memory_fraction:1.0"
        )

    def test_config_parsing_valid_false(self):
        """Test that throw_on_npumalloc_oom:False is parsed correctly."""
        torch_npu._C._npu_npuCachingAllocator_set_allocator_settings(
            "throw_on_npumalloc_oom:False"
        )

    def test_config_parsing_invalid_value(self):
        """Test that invalid throw_on_npumalloc_oom value raises error."""
        with self.assertRaises(RuntimeError):
            torch_npu._C._npu_npuCachingAllocator_set_allocator_settings(
                "throw_on_npumalloc_oom:maybe"
            )

    def test_oom_rejection_counter_in_stats(self):
        """Test that num_oom_rejections appears in memory_stats."""
        stats = torch_npu.npu.memory_stats()
        self.assertIn("num_oom_rejections", stats)

    def test_preemptive_rejection_with_low_fraction(self):
        """Test that allocations are preemptively rejected when
        per_process_memory_fraction is very low and throw_on_npumalloc_oom is True."""
        try:
            # Set a very small fraction so that any significant allocation will be rejected
            torch_npu._C._npu_npuCachingAllocator_set_allocator_settings(
                "throw_on_npumalloc_oom:True,per_process_memory_fraction:0.01"
            )
            torch_npu.npu.reset_accumulated_memory_stats()

            # 1GB allocation on a device with only 1% allowed should fail
            with self.assertRaises(torch.OutOfMemoryError):
                torch.randn(256, 1024, 1024, device="npu")

            # Verify num_oom_rejections was incremented
            stats = torch_npu.npu.memory_stats()
            self.assertGreater(stats["num_oom_rejections"], 0)
        finally:
            # Always reset to normal configuration
            torch_npu._C._npu_npuCachingAllocator_set_allocator_settings(
                "throw_on_npumalloc_oom:False,per_process_memory_fraction:1.0"
            )

    def test_normal_alloc_without_throw_on_oom(self):
        """Test that normal allocations succeed when throw_on_npumalloc_oom is False."""
        torch_npu._C._npu_npuCachingAllocator_set_allocator_settings(
            "throw_on_npumalloc_oom:False"
        )
        # Small allocation should succeed
        x = torch.randn(10, 10, device="npu")
        self.assertEqual(x.device.type, "npu")


if __name__ == "__main__":
    run_tests()
