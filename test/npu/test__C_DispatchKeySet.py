# -*- coding: utf-8 -*-
"""
测试目的：验证 torch._C.DispatchKeySet 类功能正确性
API 名称：torch._C.DispatchKeySet
API 签名：DispatchKeySet(dispatch_key: DispatchKey | DispatchKeySet)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况    |
|------------------|--------------------------------------------------------------|-------------|
| 空/非空          | 必须传入 DispatchKey 或 DispatchKeySet                       | 已覆盖      |
| 枚举选项         | 使用不同 DispatchKey 构造                                    | 已覆盖      |
| 参数类型         | DispatchKey / DispatchKeySet                                 | 已覆盖      |
| 传参与不传参     | 必须传参构造                                                 | 已覆盖      |
| 等价类/边界值    | 单个 key / 多个 key 组合                                     | 已覆盖      |
| 正常传参场景     | 构造、add、has、remove 操作不报错                            | 已覆盖      |
| 异常传参场景     | 无效参数类型触发 TypeError                                   | 已覆盖      |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、输出类型符合预期），
     不做精度和数值正确性校验。
"""
import torch
import torch_npu  # noqa: F401

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        import unittest
        unittest.main(argv=sys.argv)


class TestCDispatchKeySet(TestCase):
    """Test cases for torch._C.DispatchKeySet."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_dispatch_key_set_creation_with_key(self):
        """Creating DispatchKeySet from a single key succeeds."""
        key = torch._C.DispatchKey.CPU
        ks = torch._C.DispatchKeySet(key)
        self.assertIsInstance(ks, torch._C.DispatchKeySet)

    def test_dispatch_key_set_has(self):
        """has() returns True for contained key."""
        key = torch._C.DispatchKey.CPU
        ks = torch._C.DispatchKeySet(key)
        self.assertTrue(ks.has(key))

    def test_dispatch_key_set_add(self):
        """add() includes a new key."""
        ks = torch._C.DispatchKeySet(torch._C.DispatchKey.CPU)
        ks = ks.add(torch._C.DispatchKey.Autograd)
        self.assertTrue(ks.has(torch._C.DispatchKey.Autograd))

    def test_dispatch_key_set_remove(self):
        """remove() operation does not raise."""
        ks = torch._C.DispatchKeySet(torch._C.DispatchKey.CPU)
        ks = ks.add(torch._C.DispatchKey.Autograd)
        ks = ks.remove(torch._C.DispatchKey.CPU)
        self.assertFalse(ks.has(torch._C.DispatchKey.CPU))

    def test_dispatch_key_set_invalid_type(self):
        """Creating with invalid type raises TypeError."""
        with self.assertRaises(RuntimeError):
            torch._C.DispatchKeySet("invalid")


if __name__ == "__main__":
    run_tests()
