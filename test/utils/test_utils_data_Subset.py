# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.data.Subset 接口功能正确性
API 名称：torch.utils.data.Subset
API 签名：torch.utils.data.Subset(dataset, indices)

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | 创建子集不报错                      | 已覆盖                  |
| 长度             | 长度等于 indices 长度               | 已覆盖                  |
| getitem          | 按子集索引访问                      | 已覆盖                  |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性，不做精度和数值正确性校验。
"""
import torch
import torch_npu  # noqa: F401
from torch.utils.data import Dataset, Subset

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    import unittest
    from unittest import TestCase
    def run_tests():
        unittest.main(argv=sys.argv)


class _ListDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TestUtilsDataSubset(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_subset_len(self):
        """Verify Subset length equals indices list length."""
        ds = _ListDataset([10, 20, 30, 40, 50])
        subset = Subset(ds, [0, 2, 4])
        self.assertEqual(len(subset), 3)

    def test_subset_getitem(self):
        """Verify Subset __getitem__ maps to correct parent dataset elements."""
        ds = _ListDataset([10, 20, 30, 40, 50])
        subset = Subset(ds, [1, 3])
        self.assertEqual(subset[0], 20)
        self.assertEqual(subset[1], 40)

    def test_subset_empty_indices(self):
        """Verify Subset with empty indices has length 0."""
        ds = _ListDataset([10, 20, 30])
        subset = Subset(ds, [])
        self.assertEqual(len(subset), 0)


if __name__ == "__main__":
    run_tests()
