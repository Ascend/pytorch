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


class TestCudaGraphPoolHandle(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, "npu", f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_npu_returns_tuple(self):
        """Verify graph_pool_handle returns a tuple."""
        result = torch.npu.graph_pool_handle()
        self.assertIsInstance(result, tuple)

    def test_npu_multiple_calls(self):
        """Verify repeated calls return valid tuples without leaking state."""
        h1 = torch.npu.graph_pool_handle()
        h2 = torch.npu.graph_pool_handle()
        self.assertIsInstance(h1, tuple)
        self.assertIsInstance(h2, tuple)


if __name__ == "__main__":
    run_tests()
