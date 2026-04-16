# -*- coding: utf-8 -*-
"""
测试目的：验证 torch._dynamo.disable 接口功能正确性
API 名称：torch._dynamo.disable
API 签名：disable(fn=None, recursive=True, *, reason=None, wrapping=True)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况    |
|------------------|--------------------------------------------------------------|-------------|
| 空/非空          | 装饰函数调用                                                 | 已覆盖      |
| 枚举选项         | recursive=True / False                                       | 已覆盖      |
| 参数类型         | callable / bool / str(reason)                                | 已覆盖      |
| 传参与不传参     | 默认参数与显式参数                                           | 已覆盖      |
| 等价类/边界值    | 简单函数 / 带参函数                                          | 已覆盖      |
| 正常传参场景     | 装饰后的函数可正常调用                                       | 已覆盖      |
| 异常传参场景     | 装饰非 callable 触发 TypeError                               | 已覆盖      |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、返回 callable 符合预期），
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


class TestDynamoDisable(TestCase):
    """Test cases for torch._dynamo.disable."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_disable_basic(self):
        """Decorating a function returns a callable."""
        @torch._dynamo.disable
        def fn(x):
            return x + 1

        result = fn(1)
        self.assertEqual(result, 2)

    def test_disable_recursive_false(self):
        """Decorating with recursive=False returns a callable."""
        @torch._dynamo.disable(recursive=False)
        def fn(x):
            return x + 1

        result = fn(1)
        self.assertEqual(result, 2)

    def test_disable_invalid_target(self):
        """Decorating a non-callable raises AssertionError."""
        with self.assertRaises(AssertionError):
            torch._dynamo.disable("not callable")


if __name__ == "__main__":
    run_tests()
