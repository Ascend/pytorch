import unittest
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import freeze_rng_state

TEST_NPU_SUPPORT = False
TEST_NPU = torch.npu.is_available()
TEST_MULTINPU = TEST_NPU and torch.npu.device_count() >= 2

device = 'npu:0'
torch.npu.set_device(device)


class TestRandomSampling(TestCase):
    def test_seed(self):
        torch.seed()

    def test_manual_seed(self):
        with freeze_rng_state():
            x = torch.zeros(4, 4).float().npu()
            torch.npu.manual_seed(2)
            self.assertEqual(torch.npu.initial_seed(), 2)
            x.uniform_()
            a = torch.bernoulli(torch.full_like(x, 0.5))
            torch.npu.manual_seed(2)
            y = x.clone().uniform_()
            b = torch.bernoulli(torch.full_like(x, 0.5))
            self.assertRtolEqual(x.cpu().numpy(), y.cpu().numpy())
            self.assertRtolEqual(a.cpu().numpy(), b.cpu().numpy())
            self.assertEqual(torch.npu.initial_seed(), 2)

    @unittest.skipIf(not TEST_MULTINPU, "only one NPU detected")
    @unittest.skipIf(not TEST_NPU_SUPPORT, "NPU not support")
    def test_get_set_rng_state_all(self):
        states = torch.npu.get_rng_state_all()
        before0 = torch.npu.FloatTensor(100, device=0).normal_()
        before1 = torch.npu.FloatTensor(100, device=1).normal_()
        torch.npu.set_rng_state_all(states)
        after0 = torch.npu.FloatTensor(100, device=0).normal_()
        after1 = torch.npu.FloatTensor(100, device=1).normal_()
        self.assertEqual(before0, after0, 0)
        self.assertEqual(before1, after1, 0)

    def test_rand(self):
        out = torch.rand(2, 3, device=device)

    def test_rand_like(self):
        input1 = torch.randn((2, 3), device=device)
        out = torch.rand_like(input1, device=device)

    def test_randint(self):
        npu_output1 = torch.randint(3, 5, (3,), device=device)
        npu_output2 = torch.randint(10, (2, 2), device=device)
        npu_output3 = torch.randint(3, 10, (2, 2), device=device)

    def test_randint_like(self):
        input1 = torch.randn((2, 3), device=device)
        output = torch.randint_like(input1, high=8, device=device)

    def test_randn(self):
        output = torch.randn((2, 3), device=device)

    def test_randn_like(self):
        input1 = torch.randn((2, 3), device=device)
        output = torch.randn_like(input1, device=device)


class TestQuasiRandomSampling(TestCase):
    def test_quasirandom_sobolEngine(self):
        soboleng = torch.quasirandom.SobolEngine(dimension=5)
        output = soboleng.draw(3)


if __name__ == "__main__":
    run_tests()
