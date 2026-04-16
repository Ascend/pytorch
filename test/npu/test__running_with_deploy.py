# -*- coding: utf-8 -*-
"""
测试目的：验证 torch._running_with_deploy 接口功能正确性
API 名称：torch._running_with_deploy
API 签名：_running_with_deploy() -> bool

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况    |
|------------------|--------------------------------------------------------------|-------------|
| 空/非空          | 无参数调用                                                   | 已覆盖      |
| 枚举选项         | 无枚举选项                                                   | 未覆盖，API 无枚举参数 |
| 参数类型         | 无参数                                                       | 未覆盖，API 无参数 |
| 传参与不传参     | 仅支持无参调用                                               | 已覆盖      |
| 等价类/边界值    | 返回值恒为 False                                             | 已覆盖      |
| 正常传参场景     | 基础调用验证返回类型与值                                     | 已覆盖      |
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


class TestRunningWithDeploy(TestCase):
    """Test cases for torch._running_with_deploy."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_running_with_deploy_returns_bool(self):
        """Basic call returns a bool."""
        result = torch._running_with_deploy()
        self.assertIsInstance(result, bool)

    def test_running_with_deploy_value(self):
        """Running with deploy should return False in normal environment."""
        result = torch._running_with_deploy()
        self.assertEqual(result, False)

    def test_running_with_deploy_invalid_args(self):
        """Passing extra arguments raises TypeError."""
        with self.assertRaises(TypeError):
            torch._running_with_deploy(1)


if __name__ == "__main__":
    run_tests()
