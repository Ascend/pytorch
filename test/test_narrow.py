# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.narrow 接口功能正确性
API 名称：torch.narrow
API 签名：torch.narrow(input, dim, start, length) -> Tensor

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | length=0；输入沿 dim 的 size 为 0                            | 已覆盖                                         |
| 枚举选项         | dim 取 0、正索引、负索引；start 取非负、负索引               | 已覆盖                                         |
| 参数类型         | input 为 Tensor；dim/start/length 为 int                     | 已覆盖                                         |
| 传参与不传参     | 无默认位置参数省略场景（四参齐全）                           | 不适用                                         |
| 等价类/边界值    | 全长 narrow、高维、非连续输入、Tensor.narrow 方法形式        | 已覆盖                                         |
| 正常传参场景     | NPU 上典型 shape / dtype，输出 shape/dtype/device            | 已覆盖                                         |
| 异常传参场景     | 非法 dim、越界窗口、非法 length                              | 已覆盖                                         |
| 混合设备输入     | 单 Tensor 输入，不适用                                       | 不适用                                         |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/device/类型符合预期），
     不做精度和数值正确性校验。
"""
import torch
import torch_npu  # noqa: F401

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    import unittest
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


class TestNarrow(TestCase):
    """Functional tests for torch.narrow on NPU."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name,
            "npu",
            f"Expected device 'npu', got '{self.device_name}'",
        )
        self.device = torch.device(self.device_name)

    def test_narrow_npu_dim0(self):
        x = torch.randn(5, 4, device=self.device)
        y = torch.narrow(x, 0, 1, 3)
        self.assertEqual(y.shape, torch.Size([3, 4]))
        self.assertEqual(y.dtype, torch.float32)
        self.assertEqual(y.device.type, self.device_name)

    def test_narrow_npu_dim1(self):
        x = torch.randn(2, 6, device=self.device)
        y = torch.narrow(x, 1, 2, 3)
        self.assertEqual(y.shape, torch.Size([2, 3]))

    def test_narrow_npu_dim_negative(self):
        x = torch.randn(2, 3, 8, device=self.device)
        y = torch.narrow(x, -1, 1, 4)
        self.assertEqual(y.shape, torch.Size([2, 3, 4]))

    def test_narrow_npu_full_length(self):
        x = torch.randn(3, 4, device=self.device)
        y = torch.narrow(x, 0, 0, 3)
        self.assertEqual(y.shape, x.shape)

    def test_narrow_npu_negative_start(self):
        x = torch.randn(7, 2, device=self.device)
        y = torch.narrow(x, 0, -3, 2)
        self.assertEqual(y.shape, torch.Size([2, 2]))

    def test_narrow_npu_length_zero(self):
        x = torch.randn(4, 3, device=self.device)
        y = torch.narrow(x, 0, 2, 0)
        self.assertEqual(y.shape, torch.Size([0, 3]))

    def test_narrow_npu_tensor_method(self):
        x = torch.randn(5, 4, device=self.device)
        y = x.narrow(1, 0, 2)
        self.assertEqual(y.shape, torch.Size([5, 2]))
        self.assertEqual(y.device.type, self.device_name)

    def test_narrow_npu_high_rank(self):
        x = torch.randn(2, 3, 4, 5, 6, device=self.device)
        y = torch.narrow(x, 2, 1, 2)
        self.assertEqual(y.shape, torch.Size([2, 3, 2, 5, 6]))

    def test_narrow_npu_non_contiguous(self):
        x = torch.randn(6, 4, device=self.device).t()
        self.assertFalse(x.is_contiguous())
        y = torch.narrow(x, 0, 1, 3)
        self.assertEqual(y.shape, torch.Size([3, 6]))
        self.assertEqual(y.device.type, self.device_name)

    def test_narrow_npu_empty_source_dim(self):
        x = torch.empty(0, 3, device=self.device, dtype=torch.float32)
        y = torch.narrow(x, 0, 0, 0)
        self.assertEqual(y.shape, torch.Size([0, 3]))

    def test_narrow_npu_supported_dtypes(self):
        dtypes = [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int32,
            torch.int64,
            torch.bool,
        ]
        for dtype in dtypes:
            if dtype == torch.bool:
                x = torch.tensor([[True, False], [False, True], [True, True]], device=self.device)
            elif dtype in (torch.int32, torch.int64):
                x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=dtype, device=self.device)
            else:
                x = torch.ones(5, 2, dtype=dtype, device=self.device)
            y = torch.narrow(x, 0, 1, 2)
            self.assertEqual(y.dtype, dtype, f"dtype mismatch for {dtype}")
            self.assertEqual(y.shape, torch.Size([2, 2]))

    def test_narrow_npu_invalid_dim_raises(self):
        x = torch.randn(2, 3, device=self.device)
        with self.assertRaises((IndexError, RuntimeError)):
            torch.narrow(x, 3, 0, 1)

    def test_narrow_npu_out_of_bounds_raises(self):
        x = torch.randn(4, 3, device=self.device)
        with self.assertRaises(RuntimeError):
            torch.narrow(x, 0, 2, 5)

    def test_narrow_npu_negative_length_raises(self):
        x = torch.randn(3, 3, device=self.device)
        with self.assertRaises(RuntimeError):
            torch.narrow(x, 0, 0, -1)

    def test_narrow_cpu_baseline(self):
        x = torch.randn(5, 4)
        y = torch.narrow(x, 0, 1, 3)
        self.assertEqual(y.shape, torch.Size([3, 4]))
        self.assertEqual(y.dtype, torch.float32)

    def test_narrow_cpu_baseline_dim1(self):
        x = torch.randn(2, 6)
        y = torch.narrow(x, 1, 2, 3)
        self.assertEqual(y.shape, torch.Size([2, 3]))


if __name__ == "__main__":
    run_tests()
