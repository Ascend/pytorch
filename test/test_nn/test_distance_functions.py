import torch
import torch.nn as nn
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestDistanceFunctions(TestCase):
    def test_CosineSimilarity(self):
        input1 = torch.randn(100, 128).npu()
        input2 = torch.randn(100, 128).npu()
        cos = nn.CosineSimilarity(dim=1, eps=1e-6).npu()
        output = cos(input1, input2)
        self.assertEqual(output is not None, True)

    def test_PairwiseDistance(self):
        pdist = nn.PairwiseDistance(p=2).npu()
        input1 = torch.randn(100, 128).npu()
        input2 = torch.randn(100, 128).npu()
        output = pdist(input1, input2)
        self.assertEqual(output is not None, True)


if __name__ == "__main__":
    torch.npu.set_device(0)
    run_tests()