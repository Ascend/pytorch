# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.data.distributed.DistributedSampler 接口功能正确性
API 名称：torch.utils.data.distributed.DistributedSampler
API 签名：torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False)

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础创建         | 创建 sampler 不报错                 | 已覆盖                  |
| 单进程模拟       | num_replicas=1, rank=0             | 已覆盖                  |
| set_epoch        | set_epoch 不报错                    | 已覆盖                  |
| 迭代             | 迭代返回索引                        | 已覆盖                  |

未覆盖项及原因：
- 真实多进程：需 torch.distributed 初始化，此处单进程模拟

注意：本测试仅验证功能正确性，不做精度和数值正确性校验。
"""
import torch
import torch_npu  # noqa: F401
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    import unittest
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


class _ListDataset(Dataset):
    def __init__(self, size=10):
        self.data = list(range(size))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TestUtilsDataDistributedDistributedSampler(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_create_sampler(self):
        """Verify DistributedSampler can be created with single-process config."""
        ds = _ListDataset(10)
        sampler = DistributedSampler(ds, num_replicas=1, rank=0)
        self.assertIsInstance(sampler, DistributedSampler)

    def test_iter_single_process(self):
        """Verify single-process iteration covers all indices exactly once."""
        ds = _ListDataset(10)
        sampler = DistributedSampler(ds, num_replicas=1, rank=0)
        indices = list(sampler)
        self.assertEqual(len(indices), 10)
        self.assertEqual(set(indices), set(range(10)))

    def test_set_epoch(self):
        """Verify set_epoch does not raise."""
        ds = _ListDataset(10)
        sampler = DistributedSampler(ds, num_replicas=1, rank=0)
        sampler.set_epoch(0)

    def test_len(self):
        """Verify sampler length matches dataset size for single replica."""
        ds = _ListDataset(10)
        sampler = DistributedSampler(ds, num_replicas=1, rank=0)
        self.assertEqual(len(sampler), 10)


if __name__ == "__main__":
    run_tests()
