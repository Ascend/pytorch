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


class TestCudaGetRngStateAll(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, "npu", f"Expected device 'npu', got '{self.device_name}'"
        )

    def test_npu_returns_list(self):
        """Verify get_rng_state_all returns a list."""
        result = torch.cuda.get_rng_state_all()
        self.assertIsInstance(result, list)

    def test_npu_list_length_matches_device_count(self):
        """Verify returned list length equals device count."""
        result = torch.cuda.get_rng_state_all()
        self.assertEqual(len(result), torch.cuda.device_count())

    def test_npu_elements_are_tensors(self):
        """Verify each element in the returned list is a Tensor."""
        result = torch.cuda.get_rng_state_all()
        for state in result:
            self.assertIsInstance(state, torch.Tensor)

    def test_npu_state_changes_after_seed(self):
        """Verify RNG state list length is consistent after reseeding."""
        torch.cuda.manual_seed_all(42)
        state1 = torch.cuda.get_rng_state_all()
        torch.cuda.manual_seed_all(123)
        state2 = torch.cuda.get_rng_state_all()
        self.assertEqual(len(state1), len(state2))


if __name__ == "__main__":
    run_tests()
