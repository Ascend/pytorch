import torch
import torch.nn as nn
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestNormalizationLayers(TestCase):
    def test_BatchNorm1d(self):
        m = nn.BatchNorm1d(100).npu()
        input1 = torch.randn(20, 100).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_BatchNorm2d(self):
        m = nn.BatchNorm2d(100, affine=False).npu()
        input1 = torch.randn(20, 100, 35, 45).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_BatchNorm3d(self):
        m = nn.BatchNorm3d(100).npu()
        input1 = torch.randn(20, 100, 35, 45, 10).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_GroupNorm(self):
        m = nn.GroupNorm(3, 6).npu()
        input1 = torch.randn(20, 6, 10, 10).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_convert_sync_batchnorm(self):
        module = torch.nn.Sequential(
            torch.nn.BatchNorm1d(100),
            torch.nn.InstanceNorm1d(100)
        ).npu()
        sync_bn_module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
        children = list(sync_bn_module.children())
        self.assertEqual(children[0].__class__, torch.nn.SyncBatchNorm)
        self.assertEqual(children[1].__class__, torch.nn.InstanceNorm1d)

    def test_InstanceNorm1d(self):
        m = nn.InstanceNorm1d(100).npu()
        input1 = torch.randn(20, 100, 40).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_InstanceNorm2d(self):
        m = nn.InstanceNorm2d(100).npu()
        input1 = torch.randn(20, 100, 35, 45).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_InstanceNorm3d(self):
        m = nn.InstanceNorm3d(100).npu()
        input1 = torch.randn(20, 100, 35, 45, 10).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_LayerNorm(self):
        input1 = torch.randn(20, 5, 10, 10).npu()
        m = nn.LayerNorm(input1.size()[1:]).npu()
        output = m(input1)
        self.assertEqual(output is not None, True)

    def test_LocalResponseNorm(self):
        lrn = nn.LocalResponseNorm(2).npu()
        signal_2d = torch.randn(32, 5, 24, 24).npu()
        signal_4d = torch.randn(16, 5, 7, 7, 7, 7).npu()
        output_2d = lrn(signal_2d)
        output_4d = lrn(signal_4d)
        self.assertEqual(output_2d is not None, True)
        self.assertEqual(output_4d is not None, True)


if __name__ == "__main__":
    run_tests()
