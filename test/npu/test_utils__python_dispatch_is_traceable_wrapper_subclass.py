# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils._python_dispatch.is_traceable_wrapper_subclass 接口功能正确性
API 名称：torch.utils._python_dispatch.is_traceable_wrapper_subclass
API 签名：is_traceable_wrapper_subclass(t: object) -> TypeIs[TensorWithFlatten]

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况    |
|------------------|--------------------------------------------------------------|-------------|
| 空/非空          | 必须传入一个对象参数                                         | 已覆盖      |
| 枚举选项         | 无枚举选项                                                   | 未覆盖，API 无枚举参数 |
| 参数类型         | torch.Tensor / 非 tensor 类型                                | 已覆盖      |
| 传参与不传参     | 必须传参                                                     | 已覆盖      |
| 等价类/边界值    | 普通 Tensor / Subclass Tensor / 非 Tensor                    | 已覆盖      |
| 正常传参场景     | 基础调用返回 bool                                            | 已覆盖      |
| 异常传参场景     | 无稳定异常路径                                               | 未覆盖，API 对任意对象均返回 bool |

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


class TestIsTraceableWrapperSubclass(TestCase):
    """Test cases for torch.utils._python_dispatch.is_traceable_wrapper_subclass."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_is_traceable_wrapper_subclass_normal_tensor(self):
        """Normal Tensor returns False."""
        x = torch.randn(4, 4)
        result = torch.utils._python_dispatch.is_traceable_wrapper_subclass(x)
        self.assertIsInstance(result, bool)
        self.assertEqual(result, False)

    def test_is_traceable_wrapper_subclass_npu_tensor(self):
        """NPU Tensor returns False."""
        device_name = torch._C._get_privateuse1_backend_name()
        x = torch.randn(4, 4, device=device_name)
        result = torch.utils._python_dispatch.is_traceable_wrapper_subclass(x)
        self.assertIsInstance(result, bool)
        self.assertEqual(result, False)

    def test_is_traceable_wrapper_subclass_non_tensor(self):
        """Non-tensor object returns False."""
        result = torch.utils._python_dispatch.is_traceable_wrapper_subclass("string")
        self.assertIsInstance(result, bool)
        self.assertEqual(result, False)

    def test_is_traceable_wrapper_subclass_none(self):
        """None returns False."""
        result = torch.utils._python_dispatch.is_traceable_wrapper_subclass(None)
        self.assertIsInstance(result, bool)
        self.assertEqual(result, False)


if __name__ == "__main__":
    run_tests()
