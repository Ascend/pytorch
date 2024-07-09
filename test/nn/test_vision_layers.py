import torch
import torch.nn as nn
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestVisionLayers(TestCase):
    def test_PixelShuffle(self):
        pixel_shuffle = nn.PixelShuffle(3).npu()
        input1 = torch.randn(1, 9, 4, 4).npu()
        output = pixel_shuffle(input1)
        self.assertEqual(output is not None, True)

    def test_Upsample(self):
        input1 = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2).npu()
        m = nn.Upsample(scale_factor=2, mode='nearest').npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_UpsamplingNearest2d(self):
        input1 = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2).npu()
        m = nn.UpsamplingNearest2d(scale_factor=2).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)


if __name__ == "__main__":
    run_tests()
