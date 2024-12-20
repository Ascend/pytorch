import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestConvFunctions(TestCase):
    def test_conv1d(self):
        filters = torch.randn(33, 16, 3)
        inputs = torch.randn(20, 16, 50)
        inputs = inputs.npu()
        filters = filters.npu()
        output = F.conv1d(inputs, filters)

        self.assertExpectedInline(str(output.size()), '''torch.Size([20, 33, 48])''')

    def test_conv2d(self):
        filters = torch.randn(8, 4, 3, 3)
        inputs = torch.randn(1, 4, 5, 5)
        inputs = inputs.npu()
        filters = filters.npu()
        output = F.conv2d(inputs, filters, padding=1)

        self.assertExpectedInline(str(output.size()), '''torch.Size([1, 8, 5, 5])''')

    @SupportedDevices(['Ascend910A', 'Ascend910P'])
    def test_conv3d(self):
        filters = torch.randn(33, 16, 3, 3, 3)
        inputs = torch.randn(20, 16, 50, 10, 20)
        inputs = inputs.npu()
        filters = filters.npu()
        output = F.conv3d(inputs, filters)

        self.assertExpectedInline(str(output.size()), '''torch.Size([20, 33, 48, 8, 18])''')

    def test_conv_transpose1d(self):
        inputs = torch.randn(20, 16, 50)
        weights = torch.randn(16, 33, 5)
        inputs = inputs.npu()
        weights = weights.npu()
        output = F.conv_transpose1d(inputs, weights)

        self.assertExpectedInline(str(output.size()), '''torch.Size([20, 33, 54])''')

    def test_conv_transpose2d(self):
        inputs = torch.randn(1, 4, 5, 5)
        weights = torch.randn(4, 8, 3, 3)
        inputs = inputs.npu()
        weights = weights.npu()
        output = F.conv_transpose2d(inputs, weights, padding=1)

        self.assertExpectedInline(str(output.size()), '''torch.Size([1, 8, 5, 5])''')

    @SupportedDevices(['Ascend910A', 'Ascend910P'])
    def test_conv_transpose3d(self):
        inputs = torch.randn(20, 16, 50, 10, 20)
        weights = torch.randn(16, 33, 3, 3, 3)
        inputs = inputs.npu()
        weights = weights.npu()
        output = F.conv_transpose3d(inputs, weights)

        self.assertExpectedInline(str(output.size()), '''torch.Size([20, 33, 52, 12, 22])''')

    def test_unfold(self):
        x = torch.Tensor([[[[1, 2, 3, 4, 5],
                            [6, 7, 8, 9, 10],
                            [11, 12, 13, 14, 15],
                            [16, 17, 18, 19, 20],
                            [21, 22, 23, 24, 25]]]])
        x = F.unfold(x, (2, 2))
        self.assertExpectedInline(str(x.size()), '''torch.Size([1, 4, 16])''')

    def test_fold(self):
        inp = torch.randn(1, 3, 10, 12)
        w = torch.randn(2, 3, 4, 5)
        inp_unf = torch.nn.functional.unfold(inp, (4, 5))
        out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
        out_unf = out_unf.npu()
        out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))

        self.assertExpectedInline(str(out.size()), '''torch.Size([1, 2, 7, 8])''')


if __name__ == "__main__":
    torch.npu.set_device(0)
    run_tests()
