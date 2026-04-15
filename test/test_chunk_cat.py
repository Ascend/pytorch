# -*- coding: utf-8 -*-
"""
测试目的：验证「torch.chunk 与 torch.cat 组合」（chunk_cat 常用写法）在 NPU 上的功能正确性
API 名称：torch.chunk + torch.cat（同 dim 组合；PyTorch 无独立 torch.chunk_cat 公开符号）
API 签名：
  torch.chunk(input, chunks, dim=0) -> tuple[Tensor, ...]
  torch.cat(tensors, dim=0, *, out=None) -> Tensor
  组合：torch.cat(torch.chunk(input, chunks, dim), dim)

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 沿切分维 size 为 0 的张量                                    | 已覆盖                                         |
| 枚举选项         | dim 取 0、正索引、负索引；chunks 取 1、>1                    | 已覆盖                                         |
| 参数类型         | 与 chunk / cat 一致                                          | 已覆盖                                         |
| 传参与不传参     | chunk 省略 dim 时默认 0，再与同 dim cat                       | 已覆盖                                         |
| 等价类/边界值    | 可整除切分、不可整除切分、高维、非连续、chunks=1             | 已覆盖                                         |
| 正常传参场景     | NPU 上 round-trip 后 shape/dtype/device 与输入一致           | 已覆盖（仅结构，不比数值）                     |
| 异常传参场景     | chunk 与 cat 使用不同 dim 导致 cat shape 不兼容              | 已覆盖                                         |
| 混合设备输入     | chunk 子张量均在同一 NPU；另测 CPU 张量混入 cat 触发异常     | 已覆盖 cat 混合设备                            |

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
    import unittest
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


def _chunk_cat(x: torch.Tensor, chunks: int, dim: int) -> torch.Tensor:
    return torch.cat(torch.chunk(x, chunks, dim), dim)


class TestChunkCat(TestCase):
    """Functional tests for torch.chunk followed by torch.cat on NPU."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name,
            "npu",
            f"Expected device 'npu', got '{self.device_name}'",
        )
        self.device = torch.device(self.device_name)

    def _assert_roundtrip_structure(self, x: torch.Tensor, out: torch.Tensor) -> None:
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, x.dtype)
        self.assertEqual(out.device.type, x.device.type)

    def test_chunk_cat_npu_roundtrip_dim0(self):
        x = torch.randn(6, 4, device=self.device)
        out = _chunk_cat(x, 3, 0)
        self._assert_roundtrip_structure(x, out)

    def test_chunk_cat_npu_roundtrip_dim1(self):
        x = torch.randn(2, 8, device=self.device)
        out = _chunk_cat(x, 4, 1)
        self._assert_roundtrip_structure(x, out)

    def test_chunk_cat_npu_roundtrip_dim_negative(self):
        x = torch.randn(2, 3, 10, device=self.device)
        out = _chunk_cat(x, 2, -1)
        self._assert_roundtrip_structure(x, out)

    def test_chunk_cat_npu_roundtrip_uneven(self):
        x = torch.randn(5, 3, device=self.device)
        out = _chunk_cat(x, 2, 0)
        self._assert_roundtrip_structure(x, out)

    def test_chunk_cat_npu_high_rank(self):
        x = torch.randn(2, 3, 8, 4, 5, device=self.device)
        out = _chunk_cat(x, 2, 2)
        self._assert_roundtrip_structure(x, out)

    def test_chunk_cat_npu_non_contiguous(self):
        x = torch.randn(6, 4, device=self.device).t()
        self.assertFalse(x.is_contiguous())
        out = _chunk_cat(x, 3, 0)
        self._assert_roundtrip_structure(x, out)

    def test_chunk_cat_npu_chunks_one(self):
        x = torch.randn(4, 5, device=self.device)
        out = _chunk_cat(x, 1, 0)
        self._assert_roundtrip_structure(x, out)

    def test_chunk_cat_npu_default_chunk_dim(self):
        x = torch.randn(4, 3, device=self.device)
        parts = torch.chunk(x, 2)
        out = torch.cat(parts, 0)
        self._assert_roundtrip_structure(x, out)

    def test_chunk_cat_npu_empty_along_dim(self):
        x = torch.empty(0, 3, device=self.device, dtype=torch.float32)
        out = _chunk_cat(x, 2, 0)
        self._assert_roundtrip_structure(x, out)

    def test_chunk_cat_npu_supported_dtypes(self):
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
                x = torch.ones(6, 2, dtype=dtype, device=self.device)
            out = _chunk_cat(x, 3, 0)
            self.assertEqual(out.dtype, dtype, f"dtype mismatch for {dtype}")

    def test_chunk_cat_npu_mismatched_cat_dim_raises(self):
        x = torch.randn(5, 6, device=self.device)
        parts = torch.chunk(x, 2, dim=0)  # shapes (3, 6) and (2, 6)
        with self.assertRaises(RuntimeError):
            # NPU execution is async; force sync so the error is raised here.
            out = torch.cat(parts, dim=1)
            out.cpu()

    def test_chunk_cat_npu_cat_mixed_device_raises(self):
        x = torch.randn(4, 3, device=self.device)
        parts = list(torch.chunk(x, 2, dim=0))
        parts[1] = parts[1].cpu()
        with self.assertRaises(RuntimeError):
            torch.cat(parts, dim=0)

    def test_chunk_cat_cpu_baseline(self):
        x = torch.randn(6, 4)
        out = _chunk_cat(x, 3, 0)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, torch.float32)

    def test_chunk_cat_cpu_baseline_dim1(self):
        x = torch.randn(2, 8)
        out = _chunk_cat(x, 4, 1)
        self.assertEqual(out.shape, x.shape)


if __name__ == "__main__":
    run_tests()
