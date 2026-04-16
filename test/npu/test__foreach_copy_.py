# -*- coding: utf-8 -*-
"""
测试目的：验证 torch._foreach_copy_ 接口功能正确性
API 名称：torch._foreach_copy_
API 签名：_foreach_copy_(self, src)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况    |
|------------------|--------------------------------------------------------------|-------------|
| 空/非空          | 输入 tensor 为空与非空场景                                   | 已覆盖      |
| 枚举选项         | 无枚举选项                                                   | 未覆盖，API 无枚举参数 |
| 参数类型         | tuple / list of NPU Tensors / 不同 dtype                     | 已覆盖      |
| 传参与不传参     | 默认参数使用                                                 | 已覆盖      |
| 等价类/边界值    | 单元素 tensor / 空 tensor / 多维 tensor                      | 已覆盖      |
| 正常传参场景     | in-place 复制成功，返回 tuple of self                        | 已覆盖      |
| 异常传参场景     | 单 tensor 传入触发 TypeError                                 | 已覆盖      |

未覆盖项及原因：
- 无

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


class TestForeachCopy(TestCase):
    """Test cases for torch._foreach_copy_."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")
        self.device = torch.device(self.device_name)

    def test_foreach_copy_npu_basic(self):
        """Basic in-place copy on NPU returns self tuple."""
        dst = (torch.empty(4, 4, device=self.device),)
        src = (torch.randn(4, 4, device=self.device),)
        result = torch._foreach_copy_(dst, src)
        self.assertIsInstance(result, (tuple, list))
        self.assertIs(result[0], dst[0])
        self.assertEqual(result[0].shape, src[0].shape)
        self.assertEqual(result[0].dtype, src[0].dtype)
        self.assertEqual(result[0].device.type, self.device_name)

    def test_foreach_copy_npu_dtype_float16(self):
        """Copy with float16 tensors."""
        dst = (torch.empty(3, 3, dtype=torch.float16, device=self.device),)
        src = (torch.randn(3, 3, dtype=torch.float16, device=self.device),)
        result = torch._foreach_copy_(dst, src)
        self.assertEqual(result[0].dtype, torch.float16)

    def test_foreach_copy_npu_empty_tensor(self):
        """Copy empty tensor succeeds."""
        dst = (torch.empty(0, device=self.device),)
        src = (torch.empty(0, device=self.device),)
        result = torch._foreach_copy_(dst, src)
        self.assertEqual(result[0].shape, torch.Size([0]))

    def test_foreach_copy_npu_mixed_device(self):
        """Mixed NPU and CPU inputs copy to dst device without error."""
        dst = (torch.empty(4, device=self.device),)
        src = (torch.randn(4),)
        result = torch._foreach_copy_(dst, src)
        self.assertIsInstance(result, (tuple, list))
        self.assertEqual(result[0].device.type, self.device_name)
        self.assertEqual(result[0].shape, src[0].shape)
        self.assertEqual(result[0].dtype, src[0].dtype)

    def test_foreach_copy_invalid_single_tensor(self):
        """Passing a single tensor instead of tuple raises TypeError."""
        dst = torch.empty(4, device=self.device)
        src = torch.randn(4, device=self.device)
        with self.assertRaises(TypeError):
            torch._foreach_copy_(dst, src)

    def test_foreach_copy_cpu_baseline(self):
        """CPU baseline for in-place copy."""
        dst = (torch.empty(4, 4),)
        src = (torch.randn(4, 4),)
        result = torch._foreach_copy_(dst, src)
        self.assertIsInstance(result, (tuple, list))
        self.assertIs(result[0], dst[0])
        self.assertEqual(result[0].shape, src[0].shape)


if __name__ == "__main__":
    run_tests()
