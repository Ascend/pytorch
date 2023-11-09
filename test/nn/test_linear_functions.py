import copy
import unittest

import torch
import torch.nn.functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestLinearFunctions(TestCase):
    @unittest.skip("skip test_linear now")
    def test_linear(self):
        input1 = torch.randn(2, 3, 4)
        weight = torch.randn(3, 4)
        npu_input = copy.deepcopy(input1).npu()
        npu_weight = copy.deepcopy(weight).npu()

        cpu_output = F.linear(input1, weight)
        npu_output = F.linear(npu_input, npu_weight)

        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())
    
    @unittest.skip("skip test_bilinear now")
    def test_bilinear(self):
        input1 = torch.randn(10, 30)
        input2 = torch.randn(10, 40)
        weight = torch.randn(5, 30, 40)
        bias = torch.randn(5)

        npu_input1 = copy.deepcopy(input1).npu()
        npu_input2 = copy.deepcopy(input2).npu()
        npu_weight = copy.deepcopy(weight).npu()
        npu_bias = copy.deepcopy(bias).npu()

        cpu_output = F.bilinear(input1, input2, weight, bias)
        npu_output = F.bilinear(npu_input1, npu_input2, npu_weight, npu_bias)

        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())


if __name__ == "__main__":
    run_tests()
