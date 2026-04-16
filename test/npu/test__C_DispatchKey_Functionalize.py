# -*- coding: utf-8 -*-
"""
测试目的：验证 torch._C.DispatchKey.Functionalize 枚举值功能正确性
API 名称：torch._C.DispatchKey.Functionalize
API 签名：N/A (枚举值, DispatchKey.Functionalize)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况    |
|------------------|--------------------------------------------------------------|-------------|
| 空/非空          | 枚举值恒有定义                                               | 已覆盖      |
| 枚举选项         | 单一枚举项 Functionalize                                     | 已覆盖      |
| 参数类型         | 无参数                                                       | 未覆盖，为枚举值 |
| 传参与不传参     | 直接访问枚举值                                               | 已覆盖      |
| 等价类/边界值    | 验证为整数类型且在合理范围                                   | 已覆盖      |
| 正常传参场景     | 访问枚举值不报错                                             | 已覆盖      |
| 异常传参场景     | 无稳定异常路径                                               | 未覆盖，枚举访问无异常 |

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


class TestCDispatchKeyFunctionalize(TestCase):
    """Test cases for torch._C.DispatchKey.Functionalize."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_functionalize_is_int(self):
        """Functionalize is an integer-like enum value."""
        key = torch._C.DispatchKey.Functionalize
        self.assertIsInstance(key, torch._C.DispatchKey)

    def test_functionalize_value_non_negative(self):
        """Functionalize value is non-negative."""
        key = torch._C.DispatchKey.Functionalize
        self.assertGreaterEqual(int(key), 0)

    def test_functionalize_name(self):
        """Functionalize has expected name representation."""
        key = torch._C.DispatchKey.Functionalize
        self.assertIn('Functionalize', str(key))


if __name__ == "__main__":
    run_tests()
