import torch
import torch.nn as nn
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestRecurrentLayers(TestCase):
    def test_RNN(self):
        input1 = torch.randn(5, 3, 10).npu()
        h0 = torch.randn(2, 3, 20).npu()
        rnn = nn.RNN(10, 20, 2).npu()
        output, hn = rnn(input1, h0)
        self.assertEqual(output is not None, True)

    def test_LSTM(self):
        input1 = torch.randn(5, 3, 10).npu()
        h0 = torch.randn(2, 3, 20).npu()
        c0 = torch.randn(2, 3, 20).npu()
        rnn = nn.LSTM(10, 20, 2).npu()
        output, (hn, cn) = rnn(input1, (h0, c0))
        self.assertEqual(output is not None, True)

    def test_GRU(self):
        input1 = torch.randn(5, 3, 10).npu()
        h0 = torch.randn(2, 3, 20).npu()
        rnn = nn.GRU(10, 20, 2).npu()
        output, hn = rnn(input1, h0)
        self.assertEqual(output is not None, True)

    def test_RNNCell(self):
        input1 = torch.randn(6, 3, 10).npu()
        hx = torch.randn(3, 20).npu()
        output = []
        rnn = nn.RNNCell(10, 20).npu()
        for i in range(6):
            hx = rnn(input1[i], hx)
            output.append(hx)

    def test_LSTMCell(self):
        input1 = torch.randn(2, 3, 10).npu()
        hx = torch.randn(3, 20).npu()
        cx = torch.randn(3, 20).npu()
        output = []
        rnn = nn.LSTMCell(10, 20).npu()
        for i in range(input1.size()[0]):
            hx, cx = rnn(input1[i], (hx, cx))
            output.append(hx)
        output = torch.stack(output, dim=0)
        self.assertEqual(output is not None, True)

    def test_GRUCell(self):
        input1 = torch.randn(6, 3, 10).npu()
        hx = torch.randn(3, 20).npu()
        cx = torch.randn(3, 20).npu()
        output = []
        rnn = nn.GRUCell(10, 20).npu()
        for i in range(6):
            hx = rnn(input1[i], hx)
            output.append(hx)


if __name__ == "__main__":
    run_tests()
