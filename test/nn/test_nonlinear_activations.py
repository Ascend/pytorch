import torch
import torch.nn as nn
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests

device = 'npu:0'


class TestNonLinearActivations(TestCase):
    def test_Hardtanh(self):
        m = nn.Hardtanh(-2, 2).npu()
        input1 = torch.randn(2).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_multihead_attention(self):
        embed_dim = 128
        num_heads = 8
        sl = 10
        bs = 8
        dtype = torch.float32
        model = nn.MultiheadAttention(embed_dim, num_heads).npu().to(dtype)
        q = torch.randn(sl, bs, embed_dim, device=device, dtype=dtype)
        k = torch.randn(sl, bs, embed_dim, device=device, dtype=dtype)
        v = torch.randn(sl, bs, embed_dim, device=device, dtype=dtype)
        out = model(q, k, v)
        self.assertEqual(q.size(), out[0].size())
        self.assertEqual(dtype, out[0].dtype)

    def test_PReLU(self):
        m = nn.PReLU().npu()
        input1 = torch.randn(2).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_ReLU(self):
        m = nn.ReLU().npu()
        input1 = torch.randn(2).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_ReLU6(self):
        m = nn.ReLU6().npu()
        input1 = torch.randn(2).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_ReLU6(self):
        m = nn.ReLU6().npu()
        input1 = torch.randn(2).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_RReLU(self):
        m = nn.RReLU(0.1, 0.3).npu()
        input1 = torch.randn(2).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_CELU(self):
        m = nn.CELU().npu()
        input1 = torch.randn(2).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_GELU(self):
        m = nn.GELU().npu()
        input1 = torch.randn(2).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_Sigmoid(self):
        m = nn.Sigmoid().npu()
        input1 = torch.randn(2).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_Softplus(self):
        m = nn.Softplus().npu()
        input1 = torch.randn(2).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_Softsign(self):
        m = nn.Softsign().npu()
        input1 = torch.randn(2).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_Tanh(self):
        m = nn.Tanh().npu()
        input1 = torch.randn(2).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_Tanhshrink(self):
        m = nn.Tanhshrink().npu()
        input1 = torch.randn(2).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_Threshold(self):
        m = nn.Threshold(0.1, 20).npu()
        input1 = torch.randn(2).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)


class TestOtherNonLinearActivations(TestCase):
    def test_Softmin(self):
        m = nn.Softmin().npu()
        input1 = torch.randn(2, 3).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_Softmax(self):
        m = nn.Softmax(dim=1).npu()
        input1 = torch.randn(2, 3).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_Softmax2d(self):
        m = nn.Softmax2d().npu()
        input1 = torch.randn(2, 3, 12, 13).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_LogSoftmax(self):
        m = nn.LogSoftmax().npu()
        input1 = torch.randn(2, 3).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)


if __name__ == "__main__":
    run_tests()
