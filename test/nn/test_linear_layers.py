import torch
import torch.nn as nn
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestLinearLayers(TestCase):
    def test_Identity(self):
        m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False).npu()
        input1 = torch.randn(128, 20).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_Linear(self):
        m = nn.Linear(20, 30).npu()
        input1 = torch.randn(128, 20).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_Linear_result(self):
        m = nn.Linear(20, 30).half()
        input1 = torch.randn(128, 20).half()
        cpuout = m(input1)
        npuout = m.npu()(input1.npu())
        self.assertRtolEqual(cpuout, npuout, prec=0.005)

    def test_Bilinear(self):
        m = nn.Bilinear(20, 30, 40).npu()
        input1 = torch.randn(128, 20).npu()
        input2 = torch.randn(128, 30).npu()
        output = m(input1, input2)
        self.assertEqual(output is not None, True)


if __name__ == "__main__":
    torch.npu.set_device(0)
    run_tests()
