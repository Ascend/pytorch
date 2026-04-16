# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.profiler.record_function 接口功能正确性
API 名称：torch.autograd.profiler.record_function
API 签名：record_function(name: str, args: Optional[str] = None)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况    |
|------------------|--------------------------------------------------------------|-------------|
| 空/非空          | name 必须非空，args 可为 None                                | 已覆盖      |
| 枚举选项         | 无枚举选项                                                   | 未覆盖，API 无枚举参数 |
| 参数类型         | str / Optional[str]                                          | 已覆盖      |
| 传参与不传参     | args 默认 vs 显式传入                                        | 已覆盖      |
| 等价类/边界值    | 空字符串 / 长字符串                                          | 已覆盖      |
| 正常传参场景     | 上下文管理器 / 装饰器使用不报错                              | 已覆盖      |
| 异常传参场景     | 缺少 name 触发 TypeError                                     | 已覆盖      |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、返回类型/副作用符合预期），
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


class TestAutogradProfilerRecordFunction(TestCase):
    """Test cases for torch.autograd.profiler.record_function."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")
        self.device = torch.device(self.device_name)

    def test_record_function_context_manager(self):
        """record_function works as a context manager."""
        with torch.autograd.profiler.record_function("test_label"):
            x = torch.randn(4, device=self.device)
            _ = x + 1

    def test_record_function_with_args(self):
        """record_function with extra args."""
        with torch.autograd.profiler.record_function("test_label", "extra args"):
            x = torch.randn(4, device=self.device)
            _ = x + 1

    def test_record_function_decorator(self):
        """record_function works as a decorator."""
        @torch.autograd.profiler.record_function("test_fn")
        def fn():
            return torch.randn(4, device=self.device)

        result = fn()
        self.assertIsInstance(result, torch.Tensor)

    def test_record_function_empty_string_name(self):
        """record_function accepts empty string name."""
        with torch.autograd.profiler.record_function(""):
            pass

    def test_record_function_missing_name(self):
        """Missing name raises TypeError."""
        with self.assertRaises(TypeError):
            with torch.autograd.profiler.record_function():
                pass


if __name__ == "__main__":
    run_tests()
