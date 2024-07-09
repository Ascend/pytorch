import unittest

import torch
import numpy as np

from torch_npu.testing.testcase import TestCase, run_tests

device = 'npu:0'
torch.npu.set_device(device)


class TestUtilities(TestCase):
    @unittest.skip("Different compile parameters will cause different results")
    def test_compiled_with_cxx11_abi(self):
        output = torch.compiled_with_cxx11_abi()
        self.assertTrue(output)

    def test_result_type(self):
        self.assertEqual(torch.result_type(torch.tensor(1, dtype=torch.int, device=device), 1), torch.int)
        self.assertEqual(torch.result_type(1, torch.tensor(1, dtype=torch.int, device=device)), torch.int)
        self.assertEqual(torch.result_type(1, 1.), torch.get_default_dtype())
        self.assertEqual(torch.result_type(torch.tensor(1, device=device), 1.), torch.get_default_dtype())
        self.assertEqual(torch.result_type(torch.tensor(1, dtype=torch.long, device=device),
                         torch.tensor([1, 1], dtype=torch.int, device=device)),
                         torch.int)
        self.assertEqual(torch.result_type(torch.tensor([1., 1.], dtype=torch.float, device=device), 1.), torch.float)
        self.assertEqual(torch.result_type(torch.tensor(1., dtype=torch.float, device=device),
                         torch.tensor(1, dtype=torch.double, device=device)),
                         torch.double)

    def test_can_cast(self):
        self.assertTrue(torch.can_cast(torch.double, torch.float))
        self.assertFalse(torch.can_cast(torch.float, torch.int))

    def test_promote_types(self):
        self.assertEqual(torch.promote_types(torch.float, torch.int), torch.float)
        self.assertEqual(torch.promote_types(torch.float, torch.double), torch.double)
        self.assertEqual(torch.promote_types(torch.int, torch.uint8), torch.int)


if __name__ == "__main__":
    run_tests()
