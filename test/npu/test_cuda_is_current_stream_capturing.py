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


class TestCudaIsCurrentStreamCapturing(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, "npu", f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_npu_returns_bool(self):
        """Verify is_current_stream_capturing returns a boolean."""
        result = torch.npu.is_current_stream_capturing()
        self.assertIsInstance(result, bool)

    def test_npu_not_capturing_by_default(self):
        """Verify stream is not in capture state by default."""
        result = torch.npu.is_current_stream_capturing()
        self.assertFalse(result)


if __name__ == "__main__":
    run_tests()
