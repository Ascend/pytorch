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


class TestAmpAutocast(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, "npu", f"Expected device 'npu', got '{self.device_name}'"
        )
        self.device = torch.device(self.device_name)

    def tearDown(self):
        # Ensure autocast is disabled after each test
        torch.set_autocast_enabled(self.device_name, False)
        super().tearDown()

    def test_npu_autocast_context_manager(self):
        """Verify autocast context manager enters and exits without error."""
        with torch.amp.autocast(self.device_name):
            pass

    def test_npu_autocast_enabled_inside(self):
        """Verify autocast is enabled inside context."""
        with torch.amp.autocast(self.device_name):
            self.assertTrue(torch.is_autocast_enabled(self.device_name))

    def test_npu_autocast_disabled_outside(self):
        """Verify autocast is disabled after exiting context."""
        with torch.amp.autocast(self.device_name):
            pass
        self.assertFalse(torch.is_autocast_enabled(self.device_name))

    def test_npu_autocast_disabled_context(self):
        """Verify autocast with enabled=False keeps it disabled inside."""
        with torch.amp.autocast(self.device_name, enabled=False):
            self.assertFalse(torch.is_autocast_enabled(self.device_name))

    def test_npu_autocast_float16(self):
        """Verify autocast with float16 dtype produces valid tensor."""
        with torch.amp.autocast(self.device_name, dtype=torch.float16):
            x = torch.randn(4, 4, device=self.device)
            result = x @ x.T
            self.assertIsInstance(result, torch.Tensor)

    def test_npu_autocast_bfloat16(self):
        """Verify autocast with bfloat16 dtype produces valid tensor."""
        with torch.amp.autocast(self.device_name, dtype=torch.bfloat16):
            x = torch.randn(4, 4, device=self.device)
            result = x @ x.T
            self.assertIsInstance(result, torch.Tensor)


if __name__ == "__main__":
    run_tests()
