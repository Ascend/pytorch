# Owner(s): ["module: tests"]
import torch
import torch_npu.testing
import torch.utils.data
from torch.testing._internal.common_utils import run_tests, TestCase, IS_FBCODE, IS_REMOTE_GPU, skipIfRocm
from torch.testing._internal.common_device_type import instantiate_device_type_tests, onlyPRIVATEUSE1, largeTensorTest
import unittest

class TestPdist(TestCase):
    def test_pdist_empty(self, device):
        shape = (0, 2)
        x = torch.randn(shape, device=device)
        self.assertEqual(torch.empty(0, device=device), torch.pdist(x))

        shape = (1, 2)
        x = torch.randn(shape, device=device)
        self.assertEqual(torch.empty(0, device=device), torch.pdist(x))

        shape = (3, 0)
        x = torch.randn(shape, device=device)
        self.assertEqual(torch.zeros(3, device=device), torch.pdist(x))

    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU, "sandcastle OOM with current tpx gpu/re configuration")
    @skipIfRocm
    @onlyPRIVATEUSE1
    @largeTensorTest('32GB', device='cpu')
    @largeTensorTest('5GB', device='npu')
    def test_pdist_norm_large(self, device):
        x = torch.randn(50000, 1, dtype=torch.float32)      # 50k * 4 bytes = 200 KB
        # Will require 1249975000 float32s
        expected_cpu = torch.pdist(x, p=2)                  # ~1250M * 4 bytes = 5 GB on CPU
        actual_cpu = torch.pdist(x.to(device), p=2).cpu()         # 5 GB on GPU + 5GB on CPU
        # Workaround for large memory overhead of self.assertTrue (see #84944)
        self.assertTrue(torch.allclose(expected_cpu, actual_cpu))  # ~20GB in allclose

instantiate_device_type_tests(TestPdist, globals(), only_for='privateuse1')

if __name__ == "__main__":
    run_tests()