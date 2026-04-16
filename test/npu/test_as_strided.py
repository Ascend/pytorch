# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.as_strided 接口功能正确性
API 名称：torch.as_strided
API 签名：as_strided(input, size, stride, storage_offset=None) -> Tensor

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况    |
|------------------|--------------------------------------------------------------|-------------|
| 空/非空          | 空 tensor 与非空 tensor 输入                                 | 已覆盖      |
| 枚举选项         | 无枚举选项                                                   | 未覆盖，API 无枚举参数 |
| 参数类型         | size: Sequence[int], stride: Sequence[int], storage_offset: int | 已覆盖      |
| 传参与不传参     | storage_offset 默认 vs 显式传入                              | 已覆盖      |
| 等价类/边界值    | 0-dim / 1D / 2D / 空 tensor / 单元素                         | 已覆盖      |
| 正常传参场景     | 构造不同 stride/view 的 tensor                               | 已覆盖      |
| 异常传参场景     | size 与 stride 长度不匹配触发 RuntimeError                   | 已覆盖      |

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


class TestAsStrided(TestCase):
    """Test cases for torch.as_strided."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")
        self.device = torch.device(self.device_name)

    def test_as_strided_npu_basic(self):
        """Basic as_strided on NPU returns correct shape and data."""
        x = torch.arange(12, dtype=torch.float32, device=self.device)
        result = torch.as_strided(x, (3, 4), (4, 1))
        self.assertEqual(result.shape, torch.Size([3, 4]))
        self.assertEqual(result.device.type, self.device_name)
        self.assertEqual(result[1][2].item(), 6.0)
        self.assertEqual(result[0][0].item(), 0.0)
        self.assertEqual(result[2][3].item(), 11.0)

    def test_as_strided_npu_with_storage_offset(self):
        """as_strided with storage_offset on NPU."""
        x = torch.randn(16, device=self.device)
        result = torch.as_strided(x, (2, 3), (4, 1), storage_offset=2)
        self.assertEqual(result.shape, torch.Size([2, 3]))

    def test_as_strided_npu_empty_tensor(self):
        """as_strided with empty size on NPU."""
        x = torch.randn(4, device=self.device)
        result = torch.as_strided(x, (0,), (1,))
        self.assertEqual(result.shape, torch.Size([0]))

    def test_as_strided_npu_scalar(self):
        """as_strided to 0-dim on NPU."""
        x = torch.randn(1, device=self.device)
        result = torch.as_strided(x, (), ())
        self.assertEqual(result.shape, torch.Size([]))

    def test_as_strided_npu_dtype_preservation(self):
        """as_strided preserves dtype on NPU."""
        x = torch.randn(8, dtype=torch.float16, device=self.device)
        result = torch.as_strided(x, (2, 4), (4, 1))
        self.assertEqual(result.dtype, torch.float16)

    def test_as_strided_npu_mismatched_stride_size(self):
        """Mismatched size and stride lengths raise RuntimeError."""
        x = torch.randn(4, device=self.device)
        with self.assertRaises(RuntimeError):
            torch.as_strided(x, (2, 2), (1,))

    def test_as_strided_cpu_baseline(self):
        """CPU baseline for as_strided."""
        x = torch.randn(12)
        result = torch.as_strided(x, (3, 4), (4, 1))
        self.assertEqual(result.shape, torch.Size([3, 4]))


if __name__ == "__main__":
    run_tests()
