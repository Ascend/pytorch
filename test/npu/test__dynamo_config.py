# -*- coding: utf-8 -*-
"""
测试目的：验证 torch._dynamo.config 模块可访问性与常见属性存在性
API 名称：torch._dynamo.config
API 签名：N/A (module)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况    |
|------------------|--------------------------------------------------------------|-------------|
| 空/非空          | 模块对象本身非空                                             | 已覆盖      |
| 枚举选项         | 无枚举选项                                                   | 未覆盖，对象为模块 |
| 参数类型         | 无参数                                                       | 未覆盖，对象为模块 |
| 传参与不传参     | 无参数场景                                                   | 已覆盖      |
| 等价类/边界值    | 验证关键属性存在                                             | 已覆盖      |
| 正常传参场景     | 访问模块及属性不报错                                         | 已覆盖      |
| 异常传参场景     | 无稳定异常路径                                               | 未覆盖，模块访问无稳定异常 |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、输出类型符合预期），
     不做精度和数值正确性校验。
"""
import types
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


class TestDynamoConfig(TestCase):
    """Test cases for torch._dynamo.config module."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_dynamo_config_is_module(self):
        """torch._dynamo.config is a module."""
        self.assertIsInstance(torch._dynamo.config, types.ModuleType)

    def test_dynamo_config_has_verbose(self):
        """Module has 'verbose' attribute."""
        self.assertTrue(hasattr(torch._dynamo.config, 'verbose'))

    def test_dynamo_config_has_cache_size_limit(self):
        """Module has 'cache_size_limit' attribute."""
        self.assertTrue(hasattr(torch._dynamo.config, 'cache_size_limit'))

    def test_dynamo_config_has_suppress_errors(self):
        """Module has 'suppress_errors' attribute."""
        self.assertTrue(hasattr(torch._dynamo.config, 'suppress_errors'))

    def test_dynamo_config_has_dynamic_shapes(self):
        """Module has 'dynamic_shapes' attribute."""
        self.assertTrue(hasattr(torch._dynamo.config, 'dynamic_shapes'))


if __name__ == "__main__":
    run_tests()
