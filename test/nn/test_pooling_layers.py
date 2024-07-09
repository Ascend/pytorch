import torch
import torch.nn as nn
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests

device = 'npu:0'


class TestPoolingLayers(TestCase):
    def test_MaxPool1d(self):
        m = nn.MaxPool1d(3, stride=2)
        input1 = torch.randn(20, 16, 50)
        output = m.npu()(input1.npu())
        self.assertEqual(output is not None, True)

    def test_MaxPool2d(self):
        m = nn.MaxPool2d(3, stride=2)
        input1 = torch.randn(20, 16, 50, 32)
        output = m.npu()(input1.npu())
        self.assertEqual(output is not None, True)

    def test_MaxPool3d(self):
        m = nn.MaxPool3d(3, stride=2)
        input1 = torch.randn(20, 16, 50, 44, 31)
        output = m.npu()(input1.npu())
        self.assertEqual(output is not None, True)

    def test_MaxUnpool1d(self):
        pool = nn.MaxPool1d(2, stride=2, return_indices=True)
        unpool = nn.MaxUnpool1d(2, stride=2)
        input1 = torch.tensor([[[1., 2, 3, 4, 5, 6, 7, 8]]])
        output, indices = pool(input1)
        output = unpool.npu()(output.npu(), indices.npu())
        self.assertEqual(output is not None, True)

    def test_MaxUnpool2d(self):
        pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        unpool = nn.MaxUnpool2d(2, stride=2)
        input1 = torch.tensor([[[[1., 2, 3, 4],
                                 [5, 6, 7, 8],
                                 [9, 10, 11, 12],
                                 [13, 14, 15, 16]]]])
        output, indices = pool(input1)
        output = unpool.npu()(output.npu(), indices.npu())
        self.assertEqual(output is not None, True)

    def test_MaxUnpool3d(self):
        pool = nn.MaxPool3d(3, stride=2, return_indices=True)
        unpool = nn.MaxUnpool3d(3, stride=2)
        output, indices = pool(torch.randn(20, 16, 51, 33, 15))
        output = unpool.npu()(output.npu(), indices.npu())
        self.assertEqual(output is not None, True)

    def test_AvgPool1d(self):
        m = nn.AvgPool1d(3, stride=2).npu()
        output = m(torch.tensor([[[1., 2, 3, 4, 5, 6, 7]]], device=device))
        self.assertEqual(output is not None, True)

    def test_AvgPool2d(self):
        m = nn.AvgPool2d(3, stride=2).npu()
        output = m(torch.randn(20, 16, 50, 32, device=device))
        self.assertEqual(output is not None, True)

    def test_AvgPool3d(self):
        m = nn.AvgPool3d(3, stride=2).npu()
        output = m(input=torch.randn(20, 16, 50, 44, 31, device=device))
        self.assertEqual(output is not None, True)

    def test_LPPool1d(self):
        m = nn.LPPool1d(2, 3, stride=2).npu()
        output = m(input=torch.randn(20, 16, 50, device=device))
        self.assertEqual(output is not None, True)

    def test_LPPool2d(self):
        m = nn.LPPool2d(2, 3, stride=2).npu()
        output = m(input=torch.randn(20, 16, 50, 32, device=device))
        self.assertEqual(output is not None, True)

    def test_AdaptiveMaxPool1d(self):
        m = nn.AdaptiveMaxPool1d(4).npu()
        output = m(input=torch.randn(32, 16, 16, device=device))
        self.assertEqual(output is not None, True)

    def test_AdaptiveMaxPool2d(self):
        m = nn.AdaptiveMaxPool2d((2, 3)).npu()
        output = m(input=torch.randn(1, 3, 8, 9, device=device))
        self.assertEqual(output is not None, True)

    def test_AdaptiveAvgPool1d(self):
        m = nn.AdaptiveAvgPool1d(5).npu()
        output = m(input=torch.randn(1, 64, 8, device=device))
        self.assertEqual(output is not None, True)

    def test_AdaptiveAvgPool2d(self):
        m = nn.AdaptiveAvgPool2d((5, 7)).npu()
        output = m(input=torch.randn(1, 64, 8, 9, device=device))
        self.assertEqual(output is not None, True)

    def test_AdaptiveAvgPool3d(self):
        m = nn.AdaptiveAvgPool3d((1, 1, 1)).npu()
        output = m(input=torch.randn(1, 64, 8, 9, 10, device=device))
        self.assertEqual(output is not None, True)


if __name__ == "__main__":
    run_tests()
