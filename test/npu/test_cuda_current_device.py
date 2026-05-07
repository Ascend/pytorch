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


class TestCudaCurrentDevice(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, "npu", f"Expected device 'npu', got '{self.device_name}'"
        )
        self.device = torch.device(self.device_name)
        self._orig_device = torch.npu.current_device()

    def tearDown(self):
        torch.npu.set_device(self._orig_device)
        super().tearDown()

    def test_npu_returns_int(self):
        """Verify current_device returns an int."""
        result = torch.npu.current_device()
        self.assertIsInstance(result, int)

    def test_npu_returns_valid_index(self):
        """Verify current_device returns a valid device index."""
        result = torch.npu.current_device()
        self.assertGreaterEqual(result, 0)
        self.assertLess(result, torch.npu.device_count())

    def test_npu_after_set_device(self):
        """Verify current_device matches after set_device."""
        torch.npu.set_device(0)
        self.assertEqual(torch.npu.current_device(), 0)


if __name__ == "__main__":
    run_tests()
