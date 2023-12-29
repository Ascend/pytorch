import unittest
import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.contrib.module import LabelSmoothingCrossEntropy


class TestCrossentropy(TestCase):
    def test_npu_crossentropy_1(self):
        np.random.seed(123)
        data1 = np.random.randn(2, 10)
        x = torch.tensor(data1, dtype=torch.float32)
        data2 = np.random.randint(low=0, high=10, size=(2,))
        y = torch.tensor(data2, dtype=torch.float32)

        x = x.npu()
        y = y.npu()
        x.requires_grad = True
        m = LabelSmoothingCrossEntropy(10)
        npu_output = m(x, y)
        npu_output.backward()
        expect_cpu_xgrad = torch.tensor([[0.0112, 0.0899, 0.0440, 0.0074, 0.0186, 0.1729, 0.0029, 0.0216,
                                          0.1176, -0.4861],
                                         [0.0085, 0.0152, 0.0744, -0.4912, 0.0107, 0.0108, 0.1520, 0.1491,
                                          0.0457, 0.0246]], dtype=torch.float32)
        self.assertRtolEqual(torch.tensor(3.8078), npu_output.detach().cpu())
        self.assertRtolEqual(expect_cpu_xgrad, x.grad.cpu())

    def test_npu_crossentropy_2(self):
        np.random.seed(234)
        data1 = np.random.randn(2, 10)
        x = torch.tensor(data1, dtype=torch.float32)
        data2 = np.random.randint(low=0, high=10, size=(2,))
        y = torch.tensor(data2, dtype=torch.float32)

        x = x.npu()
        y = y.npu()
        x.requires_grad = True
        m = LabelSmoothingCrossEntropy(10, 0.1)
        npu_output = m(x, y)
        npu_output.backward()
        expect_cpu_xgrad = torch.tensor([[6.6930e-02, 5.7021e-03, 3.9844e-02, 7.4777e-02, 2.3734e-02,
                                          -4.1566e-03, 6.5645e-03, 7.5835e-02, -4.1660e-01, 1.2737e-01],
                                         [1.4708e-02, 8.3806e-02, 4.2438e-03, 1.0031e-01, -4.3430e-04,
                                          5.1546e-02, 1.3015e-01, 2.4124e-02, 2.7007e-02, -4.3545e-01]],
                                        dtype=torch.float32)
        self.assertRtolEqual(torch.tensor(3.0848), npu_output.cpu())
        self.assertRtolEqual(expect_cpu_xgrad, x.grad.cpu())


if __name__ == "__main__":
    run_tests()
