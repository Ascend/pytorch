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


class TestCudaEmptyCache(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, "npu", f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_npu_returns_none(self):
        """Verify empty_cache returns None."""
        result = torch.cuda.empty_cache()
        self.assertIsNone(result)

    def test_npu_multiple_calls(self):
        """Verify consecutive empty_cache calls do not raise."""
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_tests()
