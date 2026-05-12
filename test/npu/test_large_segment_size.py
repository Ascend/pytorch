# Owner(s): ["module: tests"]
import gc
import multiprocessing
import os

from torch_npu.testing.testcase import run_tests, TestCase

import torch


def _wrapper(func, env_config, exit_code):
    """Wrapper function to set environment variable and run test in subprocess."""
    os.environ["PYTORCH_NPU_ALLOC_CONF"] = env_config
    try:
        func()
        exit_code.value = 0
    except Exception as e:
        print(f"Exception: {e}")
        exit_code.value = 1


def run_in_subprocess(env_config: str, func, expect_error: bool = False):
    """Run test in subprocess to ensure environment variable takes effect.

    PYTORCH_NPU_ALLOC_CONF is parsed during allocator initialization. Setting the
    environment variable in the current process won't trigger re-parsing since the
    allocator is already initialized. Running in a fresh subprocess ensures the
    environment variable is parsed from scratch.
    """
    ctx = multiprocessing.get_context("spawn")
    exit_code = ctx.Value("i", -1)
    p = ctx.Process(target=_wrapper, args=(func, env_config, exit_code))
    p.start()
    p.join()

    if expect_error:
        return exit_code.value != 0
    else:
        return exit_code.value == 0


class TestLargeSegmentSize(TestCase):
    """Test large_segment_size_mb via PYTORCH_NPU_ALLOC_CONF."""

    def test_large_segment_size_default(self):
        """Default large_segment_size is 20MB.
        A 5MB allocation should reserve 20MB."""
        gc.collect()
        torch.npu.empty_cache()
        _ = torch.empty(5 * 1024 * 1024 // 4, device="npu", dtype=torch.float32)
        reserved = torch.npu.memory_reserved()
        self.assertGreaterEqual(reserved, 20 * 1024 * 1024)

    @staticmethod
    def _test_large_segment_size_via_env():
        _ = torch.empty(5 * 1024 * 1024 // 4, device="npu", dtype=torch.float32)
        reserved = torch.npu.memory_reserved()
        print(f"reserved={reserved}")
        assert reserved >= 50 * 1024 * 1024, f"Expected >= 50MB, got {reserved}"  # noqa: S101

    def test_large_segment_size_via_env(self):
        """Set large_segment_size_mb=50 via PYTORCH_NPU_ALLOC_CONF.
        Verify that a mid-range allocation reserves in 50MB granularity."""
        success = run_in_subprocess(
            "large_segment_size_mb:50", self._test_large_segment_size_via_env
        )
        self.assertTrue(success, "Subprocess failed")

    @staticmethod
    def _test_large_segment_size_mb_priority():
        _ = torch.empty(5 * 1024 * 1024 // 4, device="npu", dtype=torch.float32)
        reserved = torch.npu.memory_reserved()
        print(f"reserved={reserved}")
        assert reserved >= 100 * 1024 * 1024, f"Expected >= 100MB, got {reserved}"  # noqa: S101
        assert reserved < 150 * 1024 * 1024, f"Expected < 150MB, got {reserved}"  # noqa: S101

    def test_large_segment_size_mb_priority(self):
        """When both segment_size_mb and large_segment_size_mb are set,
        large_segment_size_mb takes priority."""
        success = run_in_subprocess(
            "expandable_segments:True,segment_size_mb:50,large_segment_size_mb:100",
            self._test_large_segment_size_mb_priority,
        )
        self.assertTrue(success, "Subprocess failed")

    @staticmethod
    def _test_invalid_large_segment_size():
        _ = torch.empty(5 * 1024 * 1024 // 4, device="npu", dtype=torch.float32)
        print("Should not reach here")

    def test_invalid_large_segment_size_rejected(self):
        """large_segment_size_mb must be > 10MB (kMinLargeAlloc)."""
        has_error = run_in_subprocess(
            "large_segment_size_mb:5",
            self._test_invalid_large_segment_size,
            expect_error=True,
        )
        self.assertTrue(has_error, "Expected error but subprocess succeeded")

    @staticmethod
    def _test_max_split_size_less_than_large_segment():
        _ = torch.empty(5 * 1024 * 1024 // 4, device="npu", dtype=torch.float32)
        print("Should not reach here")

    def test_max_split_size_less_than_large_segment_rejected(self):
        """max_split_size_mb must be >= large_segment_size_mb."""
        has_error = run_in_subprocess(
            "large_segment_size_mb:50,max_split_size_mb:30",
            self._test_max_split_size_less_than_large_segment,
            expect_error=True,
        )
        self.assertTrue(has_error, "Expected error but subprocess succeeded")


if __name__ == "__main__":
    run_tests()
