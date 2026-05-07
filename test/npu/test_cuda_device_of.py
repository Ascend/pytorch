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


class TestCudaDeviceOf(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, "npu", f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_npu_returns_device_of_object(self):
        """Verify device_of returns a device_of instance for an NPU tensor."""
        t = torch.tensor([1.0, 2.0], device=self.device_name)
        result = torch.npu.device_of(t)
        self.assertIsInstance(result, torch.npu.device_of)

    def test_npu_as_context_manager(self):
        """Verify device_of works as a context manager for device switching."""
        t = torch.tensor([1.0, 2.0], device=self.device_name)
        with torch.npu.device_of(t):
            pass


if __name__ == "__main__":
    run_tests()
