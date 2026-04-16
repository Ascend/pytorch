# -*- coding: utf-8 -*-
"""
测试目的：验证 torch._from_functional_tensor 接口功能正确性
API 名称：torch._from_functional_tensor
API 签名：_from_functional_tensor(arg0: torch.Tensor) -> torch.Tensor

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况    |
|------------------|--------------------------------------------------------------|-------------|
| 空/非空          | 必须传入 tensor 参数                                         | 已覆盖      |
| 枚举选项         | 无枚举选项                                                   | 未覆盖，API 无枚举参数 |
| 参数类型         | CPU Tensor / NPU Tensor / 不同 dtype                         | 已覆盖      |
| 传参与不传参     | 必须传参                                                     | 已覆盖      |
| 等价类/边界值    | 空 tensor / 单元素 tensor / 多维 tensor                      | 已覆盖      |
| 正常传参场景     | functional tensor 路径（eager 下难以构造，未覆盖）           | 未覆盖      |
| 异常传参场景     | 普通 tensor 触发 RuntimeError；非 tensor 触发 TypeError     | 已覆盖      |

未覆盖项及原因：
- functional tensor 正常路径：该 API 仅应在 functionalization 上下文中对 functional tensor 调用，eager 环境下构造 functional tensor 需要进入 functional mode，测试复杂度较高，暂以异常路径覆盖普通 tensor 输入行为。

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/device 符合预期），
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


class TestFromFunctionalTensor(TestCase):
    """Test cases for torch._from_functional_tensor."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")
        self.device = torch.device(self.device_name)

    def test_from_functional_tensor_cpu_raises(self):
        """Call with ordinary CPU tensor raises RuntimeError in eager mode."""
        x = torch.randn(4, 4)
        with self.assertRaises(RuntimeError):
            torch._from_functional_tensor(x)

    def test_from_functional_tensor_npu_raises(self):
        """Call with ordinary NPU tensor raises RuntimeError in eager mode."""
        x = torch.randn(4, 4, device=self.device)
        with self.assertRaises(RuntimeError):
            torch._from_functional_tensor(x)

    def test_from_functional_tensor_dtype_float16_raises(self):
        """Call with float16 NPU tensor raises RuntimeError in eager mode."""
        x = torch.randn(4, 4, dtype=torch.float16, device=self.device)
        with self.assertRaises(RuntimeError):
            torch._from_functional_tensor(x)

    def test_from_functional_tensor_empty_raises(self):
        """Call with empty NPU tensor raises RuntimeError in eager mode."""
        x = torch.randn(0, 4, device=self.device)
        with self.assertRaises(RuntimeError):
            torch._from_functional_tensor(x)

    def test_from_functional_tensor_scalar_raises(self):
        """Call with scalar NPU tensor raises RuntimeError in eager mode."""
        x = torch.tensor(3.0, device=self.device)
        with self.assertRaises(RuntimeError):
            torch._from_functional_tensor(x)

    def test_from_functional_tensor_invalid_type(self):
        """Call with non-tensor raises TypeError."""
        with self.assertRaises(TypeError):
            torch._from_functional_tensor("not a tensor")


if __name__ == "__main__":
    run_tests()
