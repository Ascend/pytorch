# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.data.Dataset 接口功能正确性
API 名称：torch.utils.data.Dataset
API 签名：torch.utils.data.Dataset (abstract base class)

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 子类化           | 可以继承并实现 __len__/__getitem__  | 已覆盖                  |
| 实例化后调用     | __len__ 和 __getitem__ 返回正确类型 | 已覆盖                  |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性，不做精度和数值正确性校验。
"""
import torch
import torch_npu  # noqa: F401
from torch.utils.data import Dataset

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    import unittest
    from unittest import TestCase
    def run_tests():
        unittest.main(argv=sys.argv)


class _SimpleDataset(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return idx


class TestUtilsDataDataset(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_subclass_len(self):
        """Verify Dataset subclass __len__ returns correct length."""
        ds = _SimpleDataset()
        self.assertEqual(len(ds), 10)

    def test_subclass_getitem(self):
        """Verify Dataset subclass __getitem__ returns correct item."""
        ds = _SimpleDataset()
        self.assertEqual(ds[0], 0)
        self.assertEqual(ds[9], 9)


if __name__ == "__main__":
    run_tests()
