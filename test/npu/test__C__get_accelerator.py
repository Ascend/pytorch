# -*- coding: utf-8 -*-
"""
测试目的：验证 torch._C._get_accelerator 接口功能正确性
API 名称：torch._C._get_accelerator
API 签名：_get_accelerator(check: Optional[bool] = None) -> torch.device

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况    |
|------------------|--------------------------------------------------------------|-------------|
| 空/非空          | 无参调用与传参调用                                           | 已覆盖      |
| 枚举选项         | check=True / False / None                                    | 已覆盖      |
| 参数类型         | bool / None                                                  | 已覆盖      |
| 传参与不传参     | 显式传入 vs 使用默认                                         | 已覆盖      |
| 等价类/边界值    | check 取 True 与 False                                       | 已覆盖      |
| 正常传参场景     | 返回 torch.device 且类型为 npu                               | 已覆盖      |
| 异常传参场景     | 传入非 bool 类型触发 TypeError                               | 已覆盖      |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、返回 device 类型符合预期），
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


class TestCGetAccelerator(TestCase):
    """Test cases for torch._C._get_accelerator."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")

    def test_get_accelerator_no_arg(self):
        """Call without arguments returns a torch.device."""
        result = torch._C._get_accelerator()
        self.assertIsInstance(result, torch.device)
        self.assertEqual(result.type, self.device_name)

    def test_get_accelerator_check_true(self):
        """Call with check=True returns a torch.device."""
        result = torch._C._get_accelerator(True)
        self.assertIsInstance(result, torch.device)
        self.assertEqual(result.type, self.device_name)

    def test_get_accelerator_check_false(self):
        """Call with check=False returns a torch.device."""
        result = torch._C._get_accelerator(False)
        self.assertIsInstance(result, torch.device)
        self.assertEqual(result.type, self.device_name)

    def test_get_accelerator_invalid_type(self):
        """Call with non-bool type raises TypeError."""
        with self.assertRaises(TypeError):
            torch._C._get_accelerator("invalid")


if __name__ == "__main__":
    run_tests()
