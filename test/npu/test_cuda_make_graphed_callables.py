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


class TestCudaMakeGraphedCallables(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, "npu", f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_npu_make_graphed_callables_callable(self):
        """Verify make_graphed_callables is a callable in torch.npu namespace."""
        self.assertTrue(callable(torch.npu.make_graphed_callables))


if __name__ == "__main__":
    run_tests()
