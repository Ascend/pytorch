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


class TestCudaDefaultStream(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, "npu", f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_npu_returns_stream(self):
        """Verify default_stream returns a Stream instance."""
        s = torch.npu.default_stream()
        self.assertIsInstance(s, torch.npu.Stream)

    def test_npu_with_device_arg(self):
        """Verify default_stream accepts a device index argument."""
        s = torch.npu.default_stream(0)
        self.assertIsInstance(s, torch.npu.Stream)


if __name__ == "__main__":
    run_tests()
