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


class TestCudaDeviceCount(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, "npu", f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_npu_returns_int(self):
        """Verify device_count returns an int."""
        result = torch.npu.device_count()
        self.assertIsInstance(result, int)

    def test_npu_positive_count(self):
        """Verify device_count is positive on NPU system."""
        result = torch.npu.device_count()
        self.assertGreater(result, 0)


if __name__ == "__main__":
    run_tests()
