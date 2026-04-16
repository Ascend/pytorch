# -*- coding: utf-8 -*-
"""
测试目的：验证 torch._dynamo.comptime.comptime.print 接口功能正确性
API 名称：torch._dynamo.comptime.comptime.print
API 签名：print(e: Any) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | None 与非空值                                                | 已覆盖                                         |
| 枚举选项         | 各种数据类型                                                   | 已覆盖                                         |
| 参数类型         | Tensor、str、int、float、bool、list、dict                    | 已覆盖                                         |
| 传参与不传参     | 有参调用（该 API 必须有参）                                  | 已覆盖                                         |
| 等价类/边界值    | 空字符串、空 tensor、单元素 tensor                           | 已覆盖                                         |
| 正常传参场景     | 直接调用打印各种类型数据                                     | 已覆盖                                         |
| 异常传参场景     | 缺少参数、多参数                                             | 已覆盖                                         |

未覆盖项及原因：
- compile/export 上下文：该 API 在 eager 模式下已充分验证，compile 模式需更复杂的测试环境

注意：本测试仅验证功能正确性（调用不报错、输出符合预期），不做数值精度校验。
      comptime.print 是 eager 模式下调试工具，直接调用即可，不需要 torch.compile。
"""

import contextlib
import io

import torch
import torch_npu

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    import unittest
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


class TestDynamoComptimePrint(TestCase):
    """Test cases for torch._dynamo.comptime.comptime.print on NPU devices."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")

    def test_comptime_print_npu_tensor(self):
        """Test comptime.print with NPU tensor and verify output contains device info."""
        tensor = torch.ones(1, device=self.device_name)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = torch._dynamo.comptime.comptime.print(tensor)

        output = stream.getvalue()
        self.assertIsNone(result)
        self.assertIn('npu', output.lower())

    def test_comptime_print_string(self):
        """Test comptime.print with string argument."""
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = torch._dynamo.comptime.comptime.print("test message")

        output = stream.getvalue()
        self.assertIsNone(result)
        self.assertIn('test message', output)

    def test_comptime_print_empty_string(self):
        """Test comptime.print with empty string (boundary value)."""
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = torch._dynamo.comptime.comptime.print("")

        self.assertIsNone(result)

    def test_comptime_print_int(self):
        """Test comptime.print with integer argument."""
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = torch._dynamo.comptime.comptime.print(42)

        output = stream.getvalue()
        self.assertIsNone(result)
        self.assertIn('42', output)

    def test_comptime_print_float(self):
        """Test comptime.print with float argument."""
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = torch._dynamo.comptime.comptime.print(3.14159)

        output = stream.getvalue()
        self.assertIsNone(result)
        self.assertIn('3.14159', output)

    def test_comptime_print_bool(self):
        """Test comptime.print with boolean argument."""
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result_true = torch._dynamo.comptime.comptime.print(True)
            result_false = torch._dynamo.comptime.comptime.print(False)

        output = stream.getvalue()
        self.assertIsNone(result_true)
        self.assertIsNone(result_false)
        self.assertIn('True', output)
        self.assertIn('False', output)

    def test_comptime_print_list(self):
        """Test comptime.print with list argument."""
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = torch._dynamo.comptime.comptime.print([1, 2, 3])

        output = stream.getvalue()
        self.assertIsNone(result)
        self.assertIn('[1, 2, 3]', output)

    def test_comptime_print_dict(self):
        """Test comptime.print with dict argument."""
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = torch._dynamo.comptime.comptime.print({"key": "value"})

        output = stream.getvalue()
        self.assertIsNone(result)
        self.assertIn('key', output)
        self.assertIn('value', output)

    def test_comptime_print_none(self):
        """Test comptime.print with None argument."""
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = torch._dynamo.comptime.comptime.print(None)

        output = stream.getvalue()
        self.assertIsNone(result)
        self.assertIn('None', output)

    def test_comptime_print_empty_tensor(self):
        """Test comptime.print with empty tensor."""
        tensor = torch.tensor([], device=self.device_name)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = torch._dynamo.comptime.comptime.print(tensor)

        self.assertIsNone(result)

    def test_comptime_print_single_element_tensor(self):
        """Test comptime.print with single element tensor."""
        tensor = torch.tensor(5.0, device=self.device_name)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = torch._dynamo.comptime.comptime.print(tensor)

        output = stream.getvalue()
        self.assertIsNone(result)
        self.assertIn('5.', output)

    def test_comptime_print_missing_argument_raises(self):
        """Test that missing argument raises TypeError."""
        with self.assertRaises(TypeError):
            torch._dynamo.comptime.comptime.print()

    def test_comptime_print_multiple_arguments_raises(self):
        """Test that multiple arguments raise TypeError."""
        with self.assertRaises(TypeError):
            torch._dynamo.comptime.comptime.print("arg1", "arg2", 123)


if __name__ == "__main__":
    run_tests()
