# Owner(s): ["module: tests"]

import torch_npu  # noqa: F401

import torch


try:
    from torch_npu.testing.testcase import run_tests, TestCase
except ImportError:
    import sys
    import unittest
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


class TestCudaMaxMemoryAllocated(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, "npu", f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_npu_returns_int(self):
        """Verify max_memory_allocated returns an int."""
        result = torch.cuda.max_memory_allocated()
        self.assertIsInstance(result, int)

    def test_npu_non_negative(self):
        """Verify reported value is non-negative."""
        result = torch.cuda.max_memory_allocated()
        self.assertGreaterEqual(result, 0)

    def test_npu_gte_memory_allocated(self):
        """Verify peak memory >= current memory_allocated."""
        max_val = torch.cuda.max_memory_allocated()
        cur_val = torch.cuda.memory_allocated()
        self.assertGreaterEqual(max_val, cur_val)


if __name__ == "__main__":
    run_tests()
