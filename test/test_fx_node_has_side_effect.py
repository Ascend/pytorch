# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.node.has_side_effect 接口功能正确性
API 名称：torch.fx.node.has_side_effect
API 签名：has_side_effect(fn: Callable[_P, _R]) -> Callable[_P, _R]

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况    |
|------------------|--------------------------------------------------------------|-------------|
| 空/非空          | 必须传入 callable 参数                                       | 已覆盖      |
| 枚举选项         | 无枚举选项                                                   | 未覆盖，API 无枚举参数 |
| 参数类型         | function / lambda / 任意对象                                 | 已覆盖      |
| 传参与不传参     | 装饰器使用                                                   | 已覆盖      |
| 等价类/边界值    | 简单函数 / lambda                                            | 已覆盖      |
| 正常传参场景     | 装饰后函数保留可调用性                                       | 已覆盖      |
| 异常传参场景     | 无稳定异常路径                                               | 未覆盖，实现层对任意对象均接受 |

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


class TestFxNodeHasSideEffect(TestCase):
    """Test cases for torch.fx.node.has_side_effect."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_has_side_effect_basic(self):
        """Decorating a function returns the same callable."""
        def fn(x):
            return x + 1

        wrapped = torch.fx.node.has_side_effect(fn)
        self.assertIs(wrapped, fn)
        self.assertEqual(fn(1), 2)

    def test_has_side_effect_lambda(self):
        """Decorating a lambda succeeds without error."""
        fn = lambda x: x + 1
        wrapped = torch.fx.node.has_side_effect(fn)
        self.assertIs(wrapped, fn)
        self.assertEqual(fn(1), 2)

    def test_has_side_effect_non_callable(self):
        """Non-callable object is accepted by current implementation (no type check)."""
        # Note: current implementation does not enforce callable type;
        # it transparently returns the input. This test documents that behavior.
        result = torch.fx.node.has_side_effect("not callable")
        self.assertEqual(result, "not callable")


if __name__ == "__main__":
    run_tests()
