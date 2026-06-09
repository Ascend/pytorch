# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.data.SequentialSampler 接口功能正确性
API 名称：torch.utils.data.SequentialSampler
API 签名：torch.utils.data.SequentialSampler(data_source)

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | 创建不报错                          | 已覆盖                  |
| 顺序迭代         | 返回 0..N-1 顺序索引               | 已覆盖                  |
| 长度             | __len__ 等于数据源长度              | 已覆盖                  |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性，不做精度和数值正确性校验。
"""
import torch
import torch_npu  # noqa: F401
from torch.utils.data import SequentialSampler

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    import unittest
    from unittest import TestCase
    def run_tests():
        unittest.main(argv=sys.argv)


class TestUtilsDataSequentialSampler(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_create_sampler(self):
        """Verify SequentialSampler can be created with a data source."""
        data = [1, 2, 3, 4, 5]
        sampler = SequentialSampler(data)
        self.assertIsInstance(sampler, SequentialSampler)

    def test_sequential_order(self):
        """Verify iteration yields indices in sequential order."""
        data = [10, 20, 30]
        sampler = SequentialSampler(data)
        indices = list(sampler)
        self.assertEqual(indices, [0, 1, 2])

    def test_len(self):
        """Verify sampler length matches data source length."""
        data = [1, 2, 3, 4, 5]
        sampler = SequentialSampler(data)
        self.assertEqual(len(sampler), 5)


if __name__ == "__main__":
    run_tests()
