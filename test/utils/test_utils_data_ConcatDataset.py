# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.data.ConcatDataset 接口功能正确性
API 名称：torch.utils.data.ConcatDataset
API 签名：torch.utils.data.ConcatDataset(datasets)

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | 拼接多个 Dataset 不报错             | 已覆盖                  |
| 长度             | 长度等于各子集之和                  | 已覆盖                  |
| getitem          | 索引访问跨数据集                    | 已覆盖                  |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性，不做精度和数值正确性校验。
"""
import torch
import torch_npu  # noqa: F401
from torch.utils.data import Dataset, ConcatDataset

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


class TestUtilsDataConcatDataset(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_concat_two_datasets(self):
        """Verify concatenated length equals sum of sub-datasets."""
        ds1 = _ListDataset([1, 2, 3])
        ds2 = _ListDataset([4, 5])
        concat = ConcatDataset([ds1, ds2])
        self.assertEqual(len(concat), 5)

    def test_getitem_across_datasets(self):
        """Verify index access spans across sub-datasets."""
        ds1 = _ListDataset([10, 20])
        ds2 = _ListDataset([30, 40])
        concat = ConcatDataset([ds1, ds2])
        self.assertEqual(concat[0], 10)
        self.assertEqual(concat[2], 30)

    def test_single_dataset(self):
        """Verify single-dataset concat preserves original length."""
        ds = _ListDataset([1, 2, 3])
        concat = ConcatDataset([ds])
        self.assertEqual(len(concat), 3)


if __name__ == "__main__":
    run_tests()
