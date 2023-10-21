import torch
import torch.nn as nn
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestConvLayers(TestCase):
    def test_conv1d(self):
        m = nn.Conv1d(16, 33, 3, stride=2)
        input1 = torch.randn(20, 16, 50)
        output = m.npu()(input1.npu())
        self.assertEqual(output is not None, True)

    def test_conv2d(self):
        m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        input1 = torch.randn(20, 16, 50, 100)
        output = m.npu()(input1.npu())
        self.assertEqual(output is not None, True)

    def test_conv3d(self):
        m = nn.Conv3d(16, 33, 3, stride=2)
        input1 = torch.randn(20, 16, 10, 50, 100)
        output = m.npu()(input1.npu())
        self.assertEqual(output is not None, True)

    def test_ConvTranspose1d(self):
        m = nn.ConvTranspose1d(4, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        input1 = torch.randn(16, 4, 8)
        output = m.npu()(input1.npu())
        self.assertEqual(output is not None, True)

    def test_ConvTranspose2d(self):
        m = nn.ConvTranspose2d(16, 33, 3, stride=2)
        input1 = torch.randn(20, 16, 50, 100)
        output = m.npu()(input1.npu())
        self.assertEqual(output is not None, True)

    def test_ConvTranspose3d(self):
        m = nn.ConvTranspose3d(16, 33, 3, stride=2)
        input1 = torch.randn(20, 16, 10, 50, 100)
        output = m.npu()(input1.npu())
        self.assertEqual(output is not None, True)

    def test_Unfold(self):
        unfold = nn.Unfold(kernel_size=(2, 3))
        input1 = torch.randn(2, 5, 3, 4)
        output = unfold.npu()(input1.npu())
        self.assertEqual(output is not None, True)


if __name__ == "__main__":
    run_tests()
