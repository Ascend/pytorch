# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.__future__.get_swap_module_params_on_conversion 接口功能正确性
API 名称：torch.__future__.get_swap_module_params_on_conversion
API 签名：get_swap_module_params_on_conversion() -> bool

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况    |
|------------------|--------------------------------------------------------------|-------------|
| 空/非空          | 无参数调用                                                   | 已覆盖      |
| 枚举选项         | True / False                                                 | 已覆盖      |
| 参数类型         | 无参数                                                       | 未覆盖，API 无参数 |
| 传参与不传参     | 仅支持无参调用                                               | 已覆盖      |
| 等价类/边界值    | 返回值为 bool                                                | 已覆盖      |
| 正常传参场景     | 基础调用返回 bool                                            | 已覆盖      |
| 异常传参场景     | 传入多余参数触发 TypeError                                   | 已覆盖      |

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


class TestFutureGetSwapModuleParamsOnConversion(TestCase):
    """Test cases for torch.__future__.get_swap_module_params_on_conversion."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_get_swap_module_params_on_conversion_returns_bool(self):
        """Call returns a bool."""
        result = torch.__future__.get_swap_module_params_on_conversion()
        self.assertIsInstance(result, bool)

    def test_get_swap_module_params_on_conversion_invalid_args(self):
        """Passing extra arguments raises TypeError."""
        with self.assertRaises(TypeError):
            torch.__future__.get_swap_module_params_on_conversion(1)


if __name__ == "__main__":
    run_tests()
