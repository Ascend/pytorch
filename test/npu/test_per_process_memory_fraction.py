# Owner(s): ["module: tests"]
import multiprocessing
import os

import torch_npu
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


class TestPerProcessMemoryFraction(TestCase):
    """Test per_process_memory_fraction via PYTORCH_NPU_ALLOC_CONF."""

    @staticmethod
    def _test_valid_fraction():
        total_memory = torch_npu.npu.get_device_properties(0).total_memory
        torch_npu.npu.empty_cache()

        application = int(total_memory * 0.2)
        try:
            torch.empty(application, dtype=torch.int8, device="npu")
            raise AssertionError("Should have raised OOM")
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise

    def test_valid_fraction_via_env(self):
        """Set per_process_memory_fraction=0.1 via PYTORCH_NPU_ALLOC_CONF.
        Verify that memory allocation is limited to 10% of total memory."""
        success = run_in_subprocess(
            "per_process_memory_fraction:0.1", self._test_valid_fraction
        )
        self.assertTrue(success, "Subprocess failed")

    @staticmethod
    def _test_fraction_one():
        total_memory = torch_npu.npu.get_device_properties(0).total_memory
        torch_npu.npu.empty_cache()

        application = int(total_memory * 0.5)
        _ = torch.empty(application, dtype=torch.int8, device="npu")

    def test_fraction_one_via_env(self):
        """Set per_process_memory_fraction=1.0 via PYTORCH_NPU_ALLOC_CONF.
        Should allow full memory usage."""
        success = run_in_subprocess(
            "per_process_memory_fraction:1.0", self._test_fraction_one
        )
        self.assertTrue(success, "Subprocess failed")

    @staticmethod
    def _test_invalid_fraction():
        x = torch.empty(1024, device="npu")

    def test_invalid_fraction_too_large_rejected(self):
        """per_process_memory_fraction must be <= 1.0."""
        has_error = run_in_subprocess(
            "per_process_memory_fraction:2.0",
            self._test_invalid_fraction,
            expect_error=True,
        )
        self.assertTrue(has_error, "Expected error but subprocess succeeded")

    def test_invalid_fraction_negative_rejected(self):
        """per_process_memory_fraction must be >= 0.0."""
        has_error = run_in_subprocess(
            "per_process_memory_fraction:-0.1",
            self._test_invalid_fraction,
            expect_error=True,
        )
        self.assertTrue(has_error, "Expected error but subprocess succeeded")

    def test_invalid_fraction_missing_value_rejected(self):
        """per_process_memory_fraction requires a value."""
        has_error = run_in_subprocess(
            "per_process_memory_fraction:",
            self._test_invalid_fraction,
            expect_error=True,
        )
        self.assertTrue(has_error, "Expected error but subprocess succeeded")

    @staticmethod
    def _test_fraction_with_other_options():
        total_memory = torch_npu.npu.get_device_properties(0).total_memory
        torch_npu.npu.empty_cache()

        application = int(total_memory * 0.4)
        _ = torch.empty(application, dtype=torch.int8, device="npu")

    def test_fraction_with_other_options(self):
        """Test per_process_memory_fraction combined with other options."""
        success = run_in_subprocess(
            "expandable_segments:True,per_process_memory_fraction:0.5",
            self._test_fraction_with_other_options,
        )
        self.assertTrue(success, "Subprocess failed")


if __name__ == "__main__":
    run_tests()
