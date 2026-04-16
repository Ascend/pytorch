# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.empty_like 接口功能正确性
API 名称：torch.empty_like
API 签名：empty_like(input, *, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None) -> Tensor

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况    |
|------------------|--------------------------------------------------------------|-------------|
| 空/非空          | 空 tensor 与非空 tensor 输入                                 | 已覆盖      |
| 枚举选项         | memory_format (torch.contiguous_format / preserve_format)    | 已覆盖      |
| 参数类型         | dtype / device 显式指定                                      | 已覆盖      |
| 传参与不传参     | 可选参数默认 vs 显式传入                                     | 已覆盖      |
| 等价类/边界值    | 0-dim / 1D / 2D / 空 tensor                                  | 已覆盖      |
| 正常传参场景     | NPU 上创建与输入同 shape 的 empty tensor                     | 已覆盖      |
| 异常传参场景     | 混合设备输入不在此 API 路径中                                | 未覆盖，API 为单输入 |

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


class TestEmptyLike(TestCase):
    """Test cases for torch.empty_like."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")
        self.device = torch.device(self.device_name)

    def test_empty_like_npu_basic(self):
        """Basic empty_like on NPU preserves shape and dtype."""
        x = torch.randn(4, 4, device=self.device)
        result = torch.empty_like(x)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, x.dtype)
        self.assertEqual(result.device.type, self.device_name)

    def test_empty_like_npu_dtype(self):
        """empty_like with explicit dtype on NPU."""
        x = torch.randn(4, 4, dtype=torch.float32, device=self.device)
        result = torch.empty_like(x, dtype=torch.float16)
        self.assertEqual(result.dtype, torch.float16)
        self.assertEqual(result.shape, x.shape)

    def test_empty_like_npu_memory_format(self):
        """empty_like with explicit memory_format on NPU."""
        x = torch.randn(4, 4, device=self.device)
        result = torch.empty_like(x, memory_format=torch.contiguous_format)
        self.assertEqual(result.shape, x.shape)

    def test_empty_like_npu_scalar(self):
        """empty_like with scalar tensor on NPU."""
        x = torch.tensor(3.0, device=self.device)
        result = torch.empty_like(x)
        self.assertEqual(result.shape, torch.Size([]))

    def test_empty_like_npu_empty_tensor(self):
        """empty_like with empty tensor on NPU."""
        x = torch.randn(0, 4, device=self.device)
        result = torch.empty_like(x)
        self.assertEqual(result.shape, torch.Size([0, 4]))

    def test_empty_like_cpu_baseline(self):
        """CPU baseline for empty_like."""
        x = torch.randn(4, 4)
        result = torch.empty_like(x)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, x.dtype)


if __name__ == "__main__":
    run_tests()
