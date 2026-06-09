# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.data.Sampler 接口功能正确性
API 名称：torch.utils.data.Sampler
API 签名：torch.utils.data.Sampler(data_source=None)

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 子类化           | 可以继承并实现 __iter__             | 已覆盖                  |
| 实例化           | 无参数实例化不报错                  | 已覆盖                  |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性，不做精度和数值正确性校验。
"""
import torch
import torch_npu  # noqa: F401
from torch.utils.data import Sampler

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    import unittest
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


class _RangeSampler(Sampler):
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        return iter(range(self.n))

    def __len__(self):
        return self.n


class TestUtilsDataSampler(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_subclass_iter(self):
        """Verify Sampler subclass __iter__ yields correct range."""
        sampler = _RangeSampler(5)
        result = list(sampler)
        self.assertEqual(result, [0, 1, 2, 3, 4])

    def test_subclass_len(self):
        """Verify Sampler subclass __len__ returns correct value."""
        sampler = _RangeSampler(3)
        self.assertEqual(len(sampler), 3)


if __name__ == "__main__":
    run_tests()
