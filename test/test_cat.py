# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.cat 接口功能正确性
API 名称：torch.cat
API 签名：torch.cat(tensors, dim=0, *, out=None) -> Tensor

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 空 tensor 参与拼接；空列表非法                               | 已覆盖 size-0 合法拼接；空列表触发异常         |
| 枚举选项         | dim 取 0、正索引、负索引                                     | 已覆盖                                         |
| 参数类型         | tensors 为 Tensor 序列；dim 为 int                          | 已覆盖                                         |
| 传参与不传参     | dim 省略默认 0；out 可选                                     | 已覆盖                                         |
| 等价类/边界值    | 单 tensor 列表、多 tensor、高维、非连续                      | 已覆盖                                         |
| 正常传参场景     | NPU 上典型 shape / dtype / out=                              | 已覆盖                                         |
| 异常传参场景     | 混合设备、拼接维 shape 不一致、空列表、非法 dim              | 已覆盖                                         |

未覆盖项及原因：
- float8_e8m0fnu / HiFloat8 在非 Ascend950 等环境由 SupportedDevices 跳过，属预期

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


class TestCat(TestCase):
    """Functional tests for torch.cat on NPU."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name,
            "npu",
            f"Expected device 'npu', got '{self.device_name}'",
        )
        self.device = torch.device(self.device_name)

    def test_cat_npu_dim0(self):
        a = torch.randn(2, 3, device=self.device)
        b = torch.randn(2, 3, device=self.device)
        out = torch.cat([a, b], dim=0)
        self.assertEqual(out.shape, torch.Size([4, 3]))
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.device.type, self.device_name)

    def test_cat_npu_dim1(self):
        a = torch.randn(2, 2, device=self.device)
        b = torch.randn(2, 3, device=self.device)
        out = torch.cat([a, b], dim=1)
        self.assertEqual(out.shape, torch.Size([2, 5]))
        self.assertEqual(out.device.type, self.device_name)

    def test_cat_npu_dim_negative(self):
        a = torch.randn(2, 3, 4, device=self.device)
        b = torch.randn(2, 3, 4, device=self.device)
        out = torch.cat([a, b], dim=-1)
        self.assertEqual(out.shape, torch.Size([2, 3, 8]))

    def test_cat_npu_three_tensors(self):
        xs = [torch.randn(1, 2, device=self.device) for _ in range(3)]
        out = torch.cat(xs, dim=0)
        self.assertEqual(out.shape, torch.Size([3, 2]))

    def test_cat_npu_empty_tensor_along_cat_dim(self):
        empty = torch.empty(0, 3, device=self.device, dtype=torch.float32)
        rest = torch.randn(2, 3, device=self.device)
        out = torch.cat([empty, rest], dim=0)
        self.assertEqual(out.shape, torch.Size([2, 3]))
        self.assertEqual(out.dtype, torch.float32)

    def test_cat_npu_high_rank(self):
        a = torch.randn(2, 3, 4, 5, 6, device=self.device)
        b = torch.randn(2, 3, 4, 5, 6, device=self.device)
        out = torch.cat([a, b], dim=2)
        self.assertEqual(out.shape, torch.Size([2, 3, 8, 5, 6]))

    def test_cat_npu_non_contiguous(self):
        a = torch.randn(4, 4, device=self.device).t()
        b = torch.randn(4, 4, device=self.device).t()
        self.assertFalse(a.is_contiguous())
        self.assertFalse(b.is_contiguous())
        out = torch.cat([a, b], dim=0)
        self.assertEqual(out.shape, torch.Size([8, 4]))
        self.assertIsInstance(out, torch.Tensor)

    def test_cat_npu_out_param(self):
        a = torch.randn(2, 3, device=self.device)
        b = torch.randn(2, 3, device=self.device)
        buffer = torch.empty(4, 3, device=self.device)
        result = torch.cat([a, b], dim=0, out=buffer)
        self.assertIs(result, buffer)
        self.assertEqual(buffer.shape, torch.Size([4, 3]))

    def test_cat_npu_default_dim(self):
        a = torch.randn(2, 3, device=self.device)
        b = torch.randn(2, 3, device=self.device)
        out_default = torch.cat([a, b])
        out_explicit = torch.cat([a, b], dim=0)
        self.assertEqual(out_default.shape, out_explicit.shape)

    def test_cat_npu_single_tensor_list(self):
        x = torch.randn(3, 4, device=self.device)
        out = torch.cat([x])
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, x.dtype)

    def test_cat_npu_supported_dtypes(self):
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
                a = torch.tensor([True, False], device=self.device)
                b = torch.tensor([False, True], device=self.device)
            elif dtype in (torch.int32, torch.int64):
                a = torch.tensor([1, 2], dtype=dtype, device=self.device)
                b = torch.tensor([3, 4], dtype=dtype, device=self.device)
            else:
                a = torch.ones(2, 2, dtype=dtype, device=self.device)
                b = torch.ones(2, 2, dtype=dtype, device=self.device)
            out = torch.cat([a, b], dim=0)
            self.assertEqual(out.dtype, dtype, f"dtype mismatch for {dtype}")

    def test_cat_npu_mixed_device_raises(self):
        a = torch.randn(2, 3, device=self.device)
        b = torch.randn(2, 3)
        with self.assertRaises(RuntimeError):
            torch.cat([a, b], dim=0)

    def test_cat_npu_incompatible_shapes_raises(self):
        a = torch.randn(2, 3, device=self.device)
        b = torch.randn(2, 4, device=self.device)
        with self.assertRaises(RuntimeError):
            # NPU execution is async; force sync so the error is raised here.
            out = torch.cat([a, b], dim=0)
            out.cpu()

    def test_cat_npu_empty_list_raises(self):
        with self.assertRaises(RuntimeError):
            torch.cat([], dim=0)

    def test_cat_npu_invalid_dim_raises(self):
        a = torch.randn(2, 3, device=self.device)
        b = torch.randn(2, 3, device=self.device)
        with self.assertRaises((IndexError, RuntimeError)):
            torch.cat([a, b], dim=2)

    def test_cat_cpu_baseline(self):
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        out = torch.cat([a, b], dim=0)
        self.assertEqual(out.shape, torch.Size([4, 3]))
        self.assertEqual(out.dtype, torch.float32)

    def test_cat_cpu_baseline_dim1(self):
        a = torch.randn(2, 2)
        b = torch.randn(2, 3)
        out = torch.cat([a, b], dim=1)
        self.assertEqual(out.shape, torch.Size([2, 5]))


if __name__ == "__main__":
    run_tests()
