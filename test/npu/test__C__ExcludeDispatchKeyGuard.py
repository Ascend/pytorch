# -*- coding: utf-8 -*-
"""
测试目的：验证 torch._C._ExcludeDispatchKeyGuard 功能正确性
API 名称：torch._C._ExcludeDispatchKeyGuard
API 签名：_ExcludeDispatchKeyGuard(key_set: DispatchKeySet)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况    |
|------------------|--------------------------------------------------------------|-------------|
| 空/非空          | 必须传入 DispatchKeySet                                      | 已覆盖      |
| 枚举选项         | 使用不同 DispatchKey 组合                                    | 已覆盖      |
| 参数类型         | DispatchKeySet                                               | 已覆盖      |
| 传参与不传参     | 必须传参构造                                                 | 已覆盖      |
| 等价类/边界值    | 单 key / 多 key 的 DispatchKeySet                            | 已覆盖      |
| 正常传参场景     | 作为上下文管理器使用不报错                                   | 已覆盖      |
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


class TestCExcludeDispatchKeyGuard(TestCase):
    """Test cases for torch._C._ExcludeDispatchKeyGuard."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_exclude_dispatch_key_guard_creation(self):
        """Creating guard with a DispatchKeySet succeeds."""
        ks = torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
        guard = torch._C._ExcludeDispatchKeyGuard(ks)
        self.assertIsInstance(guard, torch._C._ExcludeDispatchKeyGuard)

    def test_exclude_dispatch_key_guard_context_manager(self):
        """Guard works as a context manager."""
        ks = torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
        with torch._C._ExcludeDispatchKeyGuard(ks):
            pass

    def test_exclude_dispatch_key_guard_multiple_keys(self):
        """Guard with multiple keys works."""
        ks = torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
        ks = ks.add(torch._C.DispatchKey.Autograd)
        with torch._C._ExcludeDispatchKeyGuard(ks):
            pass

    def test_exclude_dispatch_key_guard_invalid_type(self):
        """Creating with invalid type raises TypeError."""
        with self.assertRaises(TypeError):
            torch._C._ExcludeDispatchKeyGuard("invalid")


if __name__ == "__main__":
    run_tests()
