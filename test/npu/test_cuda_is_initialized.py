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


class TestCudaIsInitialized(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, "npu", f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_npu_returns_bool(self):
        """Verify is_initialized returns a boolean."""
        result = torch.npu.is_initialized()
        self.assertIsInstance(result, bool)

    def test_npu_initialized_after_tensor_creation(self):
        """Verify is_initialized returns bool after NPU tensor creation."""
        _ = torch.tensor(1.0, device=self.device_name)
        self.assertIsInstance(torch.npu.is_initialized(), bool)


if __name__ == "__main__":
    run_tests()
