import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from torch_npu.contrib.module import NpuFairseqDropout, NpuCachedDropout

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class NpuMNIST(nn.Module):

    def __init__(self):
        super(NpuMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x, dropout):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = dropout(x)
        x = x.view(-1, 40)
        return x


class TestEnsembleDropout(unittest.TestCase):
    def test_EnsembleDropout(self):
        model = NpuMNIST().to("npu")
        x = torch.randn(2, 10, 16, 16).to("npu")
        NpuFairseqDropout.enable_dropout_ensemble(model)
        dropout = NpuFairseqDropout(p=0.5)
        output = model(x, dropout)

        NpuCachedDropout.enable_dropout_ensemble(model)
        dropout = NpuCachedDropout(p=0.5)
        output = model(x, dropout)


if __name__ == "__main__":
    run_tests()
