import unittest
import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.contrib.module import BiLSTM


class TestBidirectionalLstm(TestCase):

    def npu_bidirectional_lstm(self, input1):
        input1 = input1.npu()
        input1.requires_grad = True
        rnn = BiLSTM(8, 4).npu()
        input1.retain_grad()
        output = rnn(input1)
        output.backward(torch.ones(input1.size(), dtype=torch.float).npu())
        input_grad = input1.grad.cpu()
        return output.detach().cpu(), input_grad.cpu()

    def test_bidirectional_lstm(self):
        np.random.seed(123)
        data1 = np.random.randn(2, 2, 8)
        cpu_input = torch.tensor(data1, dtype=torch.float32)
        npu_input = cpu_input.npu()

        npu_output, npu_inputgrad = self.npu_bidirectional_lstm(npu_input)
        expect_cpu_output = torch.tensor([[[0.1078, 0.0723, 0.0158, 0.1555, -0.0975, 0.0867, -0.1653, -0.0974],
                                           [0.1635, 0.3083, -0.2793, 0.0748, -0.1528, 0.1982, -0.0063, 0.1506]],

                                          [[0.0743, 0.1489, -0.0777, 0.0052, -0.2747, 0.0872, -0.1923, -0.2612],
                                           [-0.1760, 0.2629, -0.2959, 0.0299, 0.3979, 0.1146, -0.1663, 0.0682]]],
                                         dtype=torch.float32)
        expect_cpu_inputgrad = torch.tensor([[[0.5643, -0.2415, -0.7003, -0.1445, 0.3078, 0.0442, -0.5876, -0.3430],
                                              [-0.1395, 0.2832, -0.3813, 0.0043, 0.0973, 0.0093, -0.2702, -0.0069]],

                                             [[-0.0751, -0.0903, -0.4377, -0.2069, 0.0934, -0.1738, -0.1134, -0.0765],
                                              [0.3855, -0.1033, -0.2906, -0.2387, -0.0037, 0.0606, -0.2736, -0.1440]]],
                                            dtype=torch.float32)
        self.assertRtolEqual(expect_cpu_output, npu_output, prec=1.e-3)
        self.assertRtolEqual(expect_cpu_inputgrad, npu_inputgrad, prec=1.e-3)


if __name__ == "__main__":
    run_tests()
