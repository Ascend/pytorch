import torch

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TorchFFTPlanCacheApiTestCase(TestCase):

    def test_max_size(self):
        torch.npu.backends.fft_plan_cache.max_size = 20
        self.assertEqual(torch.npu.backends.fft_plan_cache.max_size, 20)

    @SupportedDevices(['Ascend910B'])
    def test_size(self):
        sig = torch.randn(1, 7, dtype=torch.complex64).npu()
        torch.fft.ifft(sig)
        self.assertEqual(torch.npu.backends.fft_plan_cache.size, 1)

    @SupportedDevices(['Ascend910B'])
    def test_clear(self):
        sig = torch.randn(1, 7, dtype=torch.complex64).npu()
        torch.fft.ifft(sig)
        torch.npu.backends.fft_plan_cache.clear()
        self.assertEqual(torch.npu.backends.fft_plan_cache.size, 0)

    def test_exception(self):
        try:
            torch.npu.backends.fft_plan_cache.size = 1
        except Exception:
            return True
        return False

if __name__ == "__main__":
    run_tests()
