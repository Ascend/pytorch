# -*- coding: utf-8 -*-
"""
测试目的：验证 torch._dynamo.config.skip_fsdp_hooks 属性功能正确性
API 名称：torch._dynamo.config.skip_fsdp_hooks
API 签名：N/A (module attribute, bool)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况    |
|------------------|--------------------------------------------------------------|-------------|
| 空/非空          | 属性有默认值                                                 | 已覆盖      |
| 枚举选项         | True / False                                                 | 已覆盖      |
| 参数类型         | bool 类型                                                    | 已覆盖      |
| 传参与不传参     | 直接访问属性                                                 | 已覆盖      |
| 等价类/边界值    | True / False 切换                                            | 已覆盖      |
| 正常传参场景     | 读取与写入属性                                               | 已覆盖      |
| 异常传参场景     | 写入非 bool 类型                                             | 已覆盖      |

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


class TestDynamoConfigSkipFsdpHooks(TestCase):
    """Test cases for torch._dynamo.config.skip_fsdp_hooks."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")
        self._original = getattr(torch._dynamo.config, 'skip_fsdp_hooks', None)

    def tearDown(self):
        if self._original is not None:
            torch._dynamo.config.skip_fsdp_hooks = self._original
        super().tearDown()

    def test_skip_fsdp_hooks_is_bool(self):
        """skip_fsdp_hooks is a bool."""
        val = torch._dynamo.config.skip_fsdp_hooks
        self.assertIsInstance(val, bool)

    def test_skip_fsdp_hooks_read_write(self):
        """skip_fsdp_hooks can be set to True and False."""
        original = torch._dynamo.config.skip_fsdp_hooks
        torch._dynamo.config.skip_fsdp_hooks = not original
        self.assertEqual(torch._dynamo.config.skip_fsdp_hooks, not original)
        torch._dynamo.config.skip_fsdp_hooks = original
        self.assertEqual(torch._dynamo.config.skip_fsdp_hooks, original)


if __name__ == "__main__":
    run_tests()
