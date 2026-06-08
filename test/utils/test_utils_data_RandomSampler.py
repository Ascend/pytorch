# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.data.RandomSampler 接口功能正确性
API 名称：torch.utils.data.RandomSampler
API 签名：torch.utils.data.RandomSampler(data_source, replacement=False, num_samples=None, generator=None)

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | 创建 RandomSampler 不报错           | 已覆盖                  |
| 迭代             | 迭代返回索引列表                    | 已覆盖                  |
| 长度             | __len__ 等于数据源长度              | 已覆盖                  |
| replacement      | replacement=True/False              | 已覆盖                  |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性，不做精度和数值正确性校验。
"""
import torch
import torch_npu  # noqa: F401
from torch.utils.data import RandomSampler

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    import unittest
    from unittest import TestCase
    def run_tests():
        unittest.main(argv=sys.argv)


class TestUtilsDataRandomSampler(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_create_sampler(self):
        """Verify RandomSampler can be created with a data source."""
        data = [1, 2, 3, 4, 5]
        sampler = RandomSampler(data)
        self.assertIsInstance(sampler, RandomSampler)

    def test_iter_returns_indices(self):
        """Verify iteration returns all indices exactly once (no replacement)."""
        data = [1, 2, 3, 4, 5]
        sampler = RandomSampler(data)
        indices = list(sampler)
        self.assertEqual(len(indices), 5)
        self.assertEqual(set(indices), {0, 1, 2, 3, 4})

    def test_len(self):
        """Verify sampler length matches data source length."""
        data = [1, 2, 3]
        sampler = RandomSampler(data)
        self.assertEqual(len(sampler), 3)

    def test_replacement_true(self):
        """Verify replacement=True allows repeated indices with custom num_samples."""
        data = [1, 2, 3]
        sampler = RandomSampler(data, replacement=True, num_samples=10)
        indices = list(sampler)
        self.assertEqual(len(indices), 10)
        for idx in indices:
            self.assertIn(idx, [0, 1, 2])


if __name__ == "__main__":
    run_tests()
