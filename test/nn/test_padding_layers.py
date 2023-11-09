import torch
import torch.nn as nn
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestPaddingLayers(TestCase):
    def test_ReflectionPad2d(self):
        m = nn.ReflectionPad2d(2).npu()
        input1 = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_ReplicationPad2d(self):
        m = nn.ReplicationPad2d(2).npu()
        input1 = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_ZeroPad2d(self):
        m = nn.ZeroPad2d(2).npu()
        input1 = torch.randn(1, 1, 3, 3).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_ConstantPad1d(self):
        m = nn.ConstantPad1d(2, 3.5).npu()
        input1 = torch.randn(1, 2, 4).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_ConstantPad2d(self):
        m = nn.ConstantPad2d(2, 3.5).npu()
        input1 = torch.randn(1, 2, 2).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_ConstantPad3d(self):
        m = nn.ConstantPad3d(3, 3.5).npu()
        input1 = torch.randn(16, 3, 10, 20, 30).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)


if __name__ == "__main__":
    torch.npu.set_device(0)
    run_tests()
