import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from torch_npu.contrib.module import NpuFairseqDropout, NpuCachedDropout

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor, SupportedDevices

import torch_npu
from torch_npu.utils._error_code import ErrCode, ops_error
from torch_npu.contrib.module._ensemble_dropout import NpuPreGenDropout, _PreGenDropoutTask


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

    def test_enable_dropout_ensemble(self):
        model = NpuMNIST().to("npu")
        NpuPreGenDropout.task_dict.clear()
        NpuPreGenDropout.prob.clear()

        dropout = NpuPreGenDropout(p=0.5)
        NpuPreGenDropout.enable_dropout_ensemble(model)

        self.assertIn(0.5, NpuPreGenDropout.task_dict)
        self.assertIsNotNone(NpuPreGenDropout.dropout_stream)

    def test_unregistered_probability(self):
        NpuPreGenDropout.task_dict.clear()
        dropout = NpuPreGenDropout(p=0.3)
        x = torch.randn(2, 3, 4, 4).to("npu")
        with self.assertRaises(RuntimeError):
            dropout(x)

    def test_invalid_input_type(self):
        dropout = NpuPreGenDropout(p=0.5)
        x = "invalid_input"
        with self.assertRaises(RuntimeError):
            dropout(x)

    def test_dropout_p_zero(self):
        dropout = NpuPreGenDropout(p=0)
        x = torch.randn(2, 3, 4, 4).to("npu")
        result = dropout(x)
        self.assertTrue(torch.equal(x, result))

if __name__ == "__main__":
    run_tests()
