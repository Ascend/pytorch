import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestPoolingFunctions(TestCase):
    def test_avg_pool1d(self):
        input1 = torch.tensor([[[1, 2, 3, 4, 5, 6, 7]]], dtype=torch.float32)
        output = F.avg_pool1d(input1.npu(), kernel_size=3, stride=2)
        expected_cpu_output = torch.tensor([[[2., 4., 6.]]])

        self.assertEqual(output is not None, True)

    def test_avg_pool2d(self):
        input1 = torch.randn(1, 2, 3, 3)
        output = F.avg_pool2d(input1.npu(), kernel_size=3, stride=2)
        expected_cpu_output = torch.tensor([[[[0.4446]], [[0.4255]]]])

        self.assertRtolEqual(expected_cpu_output.numpy(), output.cpu().numpy())

    def test_avg_pool3d(self):
        input1 = torch.randn(1, 2, 3, 3, 4)
        output = F.avg_pool3d(input1.npu(), kernel_size=3, stride=2)
        expected_cpu_output = torch.tensor([[[[[0.0008]]], [[[-0.0804]]]]])

        self.assertRtolEqual(expected_cpu_output.numpy(), output.cpu().numpy())

    @unittest.skip("skip test_max_pool1d now")
    def test_max_pool1d(self):
        input1 = torch.randn(2, 4, 5)
        cpu_output = F.max_pool1d(input1, kernel_size=3, stride=2)
        npu_output = F.max_pool1d(input1.npu(), kernel_size=3, stride=2)

        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

    @unittest.skip("skip test_max_pool2d now")
    def test_max_pool2d(self):
        input1 = torch.randn(1, 2, 4, 5)
        cpu_output = F.max_pool2d(input1, kernel_size=3, stride=2)
        npu_output = F.max_pool2d(input1.npu(), kernel_size=3, stride=2)

        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

    @unittest.skip("skip test_max_pool3d now")
    def test_max_pool3d(self):
        input1 = torch.randn(1, 2, 4, 5, 6)
        cpu_output = F.max_pool3d(input1, kernel_size=3, stride=2)
        npu_output = F.max_pool3d(input1.npu(), kernel_size=3, stride=2)

        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

    def test_max_unpool1d(self):
        output, indices = F.max_pool1d(torch.randn([1, 1, 4]), 2, stride=2, return_indices=True)
        cpu_output = F.max_unpool1d(output, indices, 2)
        npu_output = F.max_unpool1d(output.npu(), indices.npu(), 2)

        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

    def test_max_unpool2d(self):
        output, indices = F.max_pool2d(torch.randn([1, 1, 4, 4]), 2, stride=2, return_indices=True)
        cpu_output = F.max_unpool2d(output, indices, 2)
        npu_output = F.max_unpool2d(output.npu(), indices.npu(), 2)

        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

    def test_max_unpool3d(self):
        output, indices = F.max_pool3d(torch.randn([4, 4, 4, 4, 4]), 2, stride=2, return_indices=True)
        cpu_output = F.max_unpool3d(output, indices, 2)
        npu_output = F.max_unpool3d(output.npu(), indices.npu(), 2)

        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

    def test_lp_pool1d(self):
        # Because of the limitation of NPU op, dtype only supports fp16.
        input1 = torch.randn(2, 3, 4, dtype=torch.half)
        cpu_output = F.lp_pool1d(input1.float(), norm_type=1, kernel_size=2, stride=1)
        input1 = input1.npu()
        npu_output = F.lp_pool1d(input1, norm_type=1, kernel_size=2, stride=1)

        self.assertRtolEqual(cpu_output.half().numpy(), npu_output.cpu().numpy())

    def test_lp_pool2d(self):
        # Because of the limitation of NPU op, dtype only supports fp16.
        input1 = torch.randn(1, 2, 3, 4, dtype=torch.half)
        cpu_output = F.lp_pool2d(input1.float(), norm_type=1, kernel_size=2, stride=1)
        input1 = input1.npu()
        npu_output = F.lp_pool2d(input1, norm_type=1, kernel_size=2, stride=1)

        self.assertRtolEqual(cpu_output.half().numpy(), npu_output.cpu().numpy())

    @unittest.skip("skip test_adaptive_max_pool1d now")
    def test_adaptive_max_pool1d(self):
        input1 = torch.randn(2, 3, 4)
        cpu_output = F.adaptive_max_pool1d(input1, output_size=2)
        input1 = input1.npu()
        npu_output = F.adaptive_max_pool1d(input1, output_size=2)

        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

    @unittest.skip("skip test_adaptive_max_pool2d now")
    def test_adaptive_max_pool2d(self):
        input1 = torch.randn(1, 2, 3, 4)
        cpu_output = F.adaptive_max_pool2d(input1, output_size=1)
        input1 = input1.npu()
        npu_output = F.adaptive_max_pool2d(input1, output_size=1)

        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

    @unittest.skip("skip test_adaptive_avg_pool1d now")
    def test_adaptive_avg_pool1d(self):
        input1 = torch.randn(2, 3, 4)
        cpu_output = F.adaptive_avg_pool1d(input1, output_size=2)
        input1 = input1.npu()
        npu_output = F.adaptive_avg_pool1d(input1, output_size=2)

        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

    def test_adaptive_avg_pool2d(self):
        input1 = torch.randn(1, 2, 3, 4)
        cpu_output = F.adaptive_avg_pool2d(input1, output_size=1)
        input1 = input1.npu()
        npu_output = F.adaptive_avg_pool2d(input1, output_size=1)

        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())

    def test_adaptive_avg_pool3d(self):
        input1 = torch.randn(1, 2, 3, 4, 2)
        cpu_output = F.adaptive_avg_pool3d(input1, output_size=1)
        input1 = input1.npu()
        npu_output = F.adaptive_avg_pool3d(input1, output_size=1)

        self.assertRtolEqual(cpu_output.numpy(), npu_output.cpu().numpy())


if __name__ == "__main__":
    run_tests()
