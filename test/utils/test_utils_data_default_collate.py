# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.data.default_collate 接口功能正确性
API 名称：torch.utils.data.default_collate
API 签名：torch.utils.data.default_collate(batch)

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况                |
|------------------|-------------------------------------|-------------------------|
| 基础调用         | 合并 tensor 列表不报错              | 已覆盖                  |
| 返回类型         | 返回 Tensor                         | 已覆盖                  |
| shape            | batch 维度在第 0 维                 | 已覆盖                  |
| 空列表           | 空列表应报错或返回空                | 已覆盖                  |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性，不做精度和数值正确性校验。
"""
import torch
import torch_npu  # noqa: F401
from torch.utils.data import default_collate

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    import unittest
    from unittest import TestCase
    def run_tests():
        unittest.main(argv=sys.argv)


class TestUtilsDataDefaultCollate(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_collate_tensors(self):
        """Verify default_collate stacks tensors into a batched Tensor."""
        batch = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        result = default_collate(batch)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, torch.Size([2, 2]))

    def test_collate_float_tensors(self):
        """Verify collated float tensors have correct batch shape."""
        batch = [torch.randn(3) for _ in range(4)]
        result = default_collate(batch)
        self.assertEqual(result.shape, torch.Size([4, 3]))

    def test_collate_dicts(self):
        """Verify default_collate merges list of dicts into dict of batched tensors."""
        batch = [{'a': torch.tensor(1), 'b': torch.tensor(2)},
                 {'a': torch.tensor(3), 'b': torch.tensor(4)}]
        result = default_collate(batch)
        self.assertIsInstance(result, dict)
        self.assertIn('a', result)
        self.assertIn('b', result)

    def test_collate_empty_raises(self):
        """Verify empty list raises an error."""
        with self.assertRaises(IndexError):
            default_collate([])


if __name__ == "__main__":
    run_tests()
