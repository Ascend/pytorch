import os
import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SkipIfNotGteCANNVersion


class TestStrongConsistency(TestCase):
    def test_are_consistent_algorithms_enable(self):
        self.assertEqual(torch_npu.npu.are_consistent_algorithms_enable(), True)

    @SkipIfNotGteCANNVersion("9.0.0")
    def test_npu_matmul(self):
        torch.set_default_device('npu')

        B = 2048
        D = 4096
        R = 1000
        a = torch.linspace(-R, R, B * D).reshape(B, D)
        b = torch.linspace(-R, R, D * D).reshape(D, D)
        out1 = torch.mm(a[:1], b)
        out2 = torch.mm(a, b)[:1]
        self.assertEqual(np.array_equal(out1.cpu().numpy().view(np.uint32), out2.cpu().numpy().view(np.uint32)), True)


if __name__ == "__main__":
    torch_npu.npu.use_consistent_algorithms(True)
    run_tests()