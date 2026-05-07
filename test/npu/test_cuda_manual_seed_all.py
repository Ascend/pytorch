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


class TestCudaManualSeedAll(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, "npu", f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_npu_returns_none(self):
        """Verify manual_seed_all returns None on success."""
        result = torch.cuda.manual_seed_all(42)
        self.assertIsNone(result)

    def test_npu_seed_zero(self):
        """Verify seed=0 (boundary value) does not raise."""
        torch.cuda.manual_seed_all(0)

    def test_npu_large_seed(self):
        """Verify large seed value (2**31-1) does not raise."""
        torch.cuda.manual_seed_all(2**31 - 1)


if __name__ == "__main__":
    run_tests()
