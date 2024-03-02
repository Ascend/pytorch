import copy
import unittest

import torch
import torch.nn as nn
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestElu(TestCase):
    def test_elu(self):
        m = nn.ELU()
        input1 = torch.randn(3, 4)
        npuout = m.npu()(input1.npu())
        cpuout = m(input1)
        self.assertRtolEqual(cpuout, npuout)

if __name__ == "__main__":
    run_tests()
