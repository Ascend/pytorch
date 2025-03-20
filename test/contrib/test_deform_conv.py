import unittest
import numpy as np
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.contrib.module import DCNv2


class TestDeformConv(TestCase):
    def test_npu_deform_conv_1(self):
        np.random.seed(226)
        data1 = np.random.randn(2, 2, 3, 3)
        x = torch.tensor(data1, dtype=torch.float32)
        model = DCNv2(2, 2, 3, 2, 1)

        x = x.npu()
        x.requires_grad = True
        model = model.npu()

        output = model(x)
        output.sum().backward()
        expect_cpu_output = torch.tensor([[[[0.0077, 0.0699],
                                            [0.1435, -0.0298]],

                                           [[0.0494, -0.1641],
                                            [-0.4070, -0.0126]]],

                                          [[[0.2375, -0.3983],
                                            [-0.3438, 0.2228]],

                                           [[-0.0948, -0.1294],
                                            [-0.1455, -0.1956]]]], dtype=torch.float32)
        expect_cpu_xgrad = torch.tensor([[[[0.0355, 0.1740, 0.0355],
                                           [0.1678, 0.0109, 0.1678],
                                           [0.0355, 0.1740, 0.0355]],

                                          [[-0.1422, -0.2028, -0.1422],
                                           [-0.0641, 0.2660, -0.0641],
                                           [-0.1422, -0.2028, -0.1422]]],

                                         [[[0.0355, 0.1740, 0.0355],
                                           [0.1678, 0.0109, 0.1678],
                                           [0.0355, 0.1740, 0.0355]],

                                          [[-0.1422, -0.2028, -0.1422],
                                           [-0.0641, 0.2660, -0.0641],
                                           [-0.1422, -0.2028, -0.1422]]]], dtype=torch.float32)
        self.assertRtolEqual(expect_cpu_output, output.detach().cpu(), prec=1.e-3)
        self.assertRtolEqual(expect_cpu_xgrad, x.grad.cpu(), prec=1.e-3)

    def test_npu_deform_conv_2(self):
        np.random.seed(546)
        data1 = np.random.randn(2, 2, 5, 5)
        x = torch.tensor(data1, dtype=torch.float32)
        model = DCNv2(2, 2, 3, 2, 1)

        x = x.npu()
        x.requires_grad = True
        model = model.npu()

        output = model(x)
        output.sum().backward()
        expect_cpu_output = torch.tensor([[[[-0.1421, -0.0099, -0.2894],
                                            [-0.0360, -0.4846, 0.2676],
                                            [-0.2791, 0.0011, -0.0440]],

                                           [[0.0051, -0.1212, 0.0374],
                                            [-0.0317, 0.0370, -0.0739],
                                            [0.1564, 0.0738, -0.2175]]],

                                          [[[-0.3272, -0.0703, 0.1305],
                                            [-0.0163, 0.1823, -0.0286],
                                            [-0.5419, 0.1746, 0.1011]],

                                           [[-0.0256, 0.6328, 0.0427],
                                            [-0.4237, 0.0432, 0.2582],
                                            [0.3830, -0.0413, -0.0663]]]], dtype=torch.float32)
        expect_cpu_xgrad = torch.tensor([[[[0.0355, 0.1740, 0.0355, 0.1740, 0.0355],
                                           [0.1678, 0.0109, 0.1678, 0.0109, 0.1678],
                                           [0.0355, 0.1740, 0.0355, 0.1740, 0.0355],
                                           [0.1678, 0.0109, 0.1678, 0.0109, 0.1678],
                                           [0.0355, 0.1740, 0.0355, 0.1740, 0.0355]],

                                          [[-0.1422, -0.2028, -0.1422, -0.2028, -0.1422],
                                           [-0.0641, 0.2660, -0.0641, 0.2660, -0.0641],
                                           [-0.1422, -0.2028, -0.1422, -0.2028, -0.1422],
                                           [-0.0641, 0.2660, -0.0641, 0.2660, -0.0641],
                                           [-0.1422, -0.2028, -0.1422, -0.2028, -0.1422]]],

                                         [[[0.0355, 0.1740, 0.0355, 0.1740, 0.0355],
                                           [0.1678, 0.0109, 0.1678, 0.0109, 0.1678],
                                           [0.0355, 0.1740, 0.0355, 0.1740, 0.0355],
                                           [0.1678, 0.0109, 0.1678, 0.0109, 0.1678],
                                           [0.0355, 0.1740, 0.0355, 0.1740, 0.0355]],

                                          [[-0.1422, -0.2028, -0.1422, -0.2028, -0.1422],
                                           [-0.0641, 0.2660, -0.0641, 0.2660, -0.0641],
                                           [-0.1422, -0.2028, -0.1422, -0.2028, -0.1422],
                                           [-0.0641, 0.2660, -0.0641, 0.2660, -0.0641],
                                           [-0.1422, -0.2028, -0.1422, -0.2028, -0.1422]]]], dtype=torch.float32)
        self.assertRtolEqual(expect_cpu_output, output.detach().cpu(), prec=1.e-3)
        self.assertRtolEqual(expect_cpu_xgrad, x.grad.cpu(), prec=1.e-3)


if __name__ == "__main__":
    run_tests()
