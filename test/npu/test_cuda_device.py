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


class TestCudaDevice(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, "npu", f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_npu_returns_device_object(self):
        """Verify torch.npu.device returns a device instance."""
        result = torch.npu.device(0)
        self.assertIsInstance(result, torch.npu.device)

    def test_npu_as_context_manager(self):
        """Verify torch.npu.device works as a context manager for device switching."""
        with torch.npu.device(0):
            pass

    def test_npu_with_int_arg(self):
        """Verify device accepts an integer device index."""
        d = torch.npu.device(0)
        self.assertIsNotNone(d)

    def test_npu_with_torch_device_arg(self):
        """Verify device accepts a torch.device object."""
        d = torch.npu.device(torch.device("npu", 0))
        self.assertIsNotNone(d)


if __name__ == "__main__":
    run_tests()
