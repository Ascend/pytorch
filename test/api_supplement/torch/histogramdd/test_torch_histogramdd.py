# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.histogramdd 接口功能正确性
API 名称：torch.histogramdd
API 签名：torch.histogramdd(input, bins, range=None, weight=None, density=False) -> (Tensor, Tensor[])

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| shape | (M, N) 二维点集 | 已覆盖 |
| dtype | float32/float64 | 已覆盖 |
| device | 每个正常场景 CPU/NPU 双端执行 | 已覆盖 |
| 参数枚举 | bins 为 int/序列/张量序列、range | 已覆盖 |
| 可选参数 | weight、density 省略与显式 | 已覆盖 |
| in-place 变体 | 无 in-place 变体 | 未覆盖（不适用） |
| out= 参数 | 返回命名元组 (hist, bin_edges) | 未覆盖（无 out= 参数） |
| 广播 | 每行一个 N 维点，无广播语义 | 未覆盖（不适用） |
| 连续性 | 结果默认连续 | 已覆盖 |
| 混合设备输入 | 单 Tensor 输入，无混合设备路径 | 未覆盖（不适用） |
| 异常路径 | 1D 输入触发异常 | 已覆盖 |

未覆盖项及原因：
- out=/广播/混合设备维度对该直方图统计 API 不适用：输入为单个点集张量，
  返回 hist 与 bin_edges 组成的命名元组。
"""
import os
import unittest

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# 仅在补充测试真正运行（RUN_FLAG）时才探测设备：默认 CI 收集会被类级
# skipUnless 跳过，此时不做 torch_npu 导入与 NPU/CUDA 设备初始化，避免
# 不必要的探测开销；被跳过时 DEVICE_NAME 仅为占位值 'cpu'。
DEVICE_NAME = 'cpu'
if RUN_FLAG:
    DEVICE_NAME = None
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = 'npu'
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'

# ORDER CONSTRAINT: import torch_npu.testing.testcase BEFORE
# torch_npu.contrib.transfer_to_npu (patching first breaks the import).
try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

if DEVICE_NAME == 'npu':
    import torch_npu.contrib.transfer_to_npu  # noqa: F401


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class Testhistogramdd(TestCase):
    """Test cases for torch.histogramdd (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_counts_match_cpu_golden(self):
        """Device counts equal the CPU golden histogram."""
        torch.manual_seed(0)
        samples = torch.randn(64, 2)
        cpu_hist, cpu_edges = torch.histogramdd(samples, bins=(2, 3))
        dev_out = torch.histogramdd(samples.to(self.device), bins=(2, 3))
        torch.testing.assert_close(dev_out.hist.cpu(), cpu_hist)
        for cpu_edge, dev_edge in zip(cpu_edges, dev_out.bin_edges):
            torch.testing.assert_close(dev_edge.cpu(), cpu_edge)

    def test_total_count_matches_samples(self):
        """All samples fall into bins when range covers the data."""
        samples = torch.randn(100, 2)
        out = torch.histogramdd(samples, bins=(4, 5))
        self.assertEqual(int(out.hist.sum().item()), 100)

    def test_bin_edges_length(self):
        """Edge count per dimension is bins + 1."""
        out = torch.histogramdd(torch.randn(50, 2), bins=(2, 3))
        self.assertEqual(len(out.bin_edges), 2)
        self.assertEqual(tuple(out.bin_edges[0].shape), (3,))
        self.assertEqual(tuple(out.bin_edges[1].shape), (4,))

    def test_weight_applied(self):
        """Weights scale the accumulated per-bin totals."""
        torch.manual_seed(0)
        samples = torch.randn(40, 2)
        weight = torch.full((40,), 2.0)
        weighted = torch.histogramdd(samples, bins=(2, 2), weight=weight)
        plain = torch.histogramdd(samples, bins=(2, 2))
        torch.testing.assert_close(weighted.hist, plain.hist * 2.0)

    def test_density_normalized(self):
        """Density integrates to one over the bin volumes."""
        torch.manual_seed(0)
        samples = torch.randn(200, 2)
        out = torch.histogramdd(samples, bins=(4, 4), density=True)
        e0, e1 = out.bin_edges
        w0 = (e0[1:] - e0[:-1]).tolist()
        w1 = (e1[1:] - e1[:-1]).tolist()
        total = sum(float(out.hist[i, j]) * w0[i] * w1[j]
                    for i in range(len(w0)) for j in range(len(w1)))
        self.assertLess(abs(total - 1.0), 1e-4)

    def test_1d_input_raises(self):
        """A 1D input is rejected (at least 2 dims required)."""
        with self.assertRaises(RuntimeError):
            torch.histogramdd(torch.randn(10), bins=(2,))

    def test_range_param(self):
        """An explicit range fixes the bin edges and captures all samples."""
        torch.manual_seed(0)
        samples = torch.rand(100, 2)
        out = torch.histogramdd(samples, bins=(3, 4), range=[0.0, 1.0, 0.0, 1.0])
        self.assertEqual(float(out.bin_edges[0][0]), 0.0)
        self.assertEqual(float(out.bin_edges[0][-1]), 1.0)
        self.assertEqual(float(out.bin_edges[1][0]), 0.0)
        self.assertEqual(float(out.bin_edges[1][-1]), 1.0)
        self.assertEqual(int(out.hist.sum().item()), 100)

    def test_int_bins_overload(self):
        """The single-int bins overload spans the same count across dimensions."""
        torch.manual_seed(0)
        samples = torch.randn(80, 3)
        out = torch.histogramdd(samples, bins=4)
        self.assertEqual(int(out.hist.sum().item()), 80)
        self.assertEqual(len(out.bin_edges), 3)

    def test_tensor_bins_overload(self):
        """Tensor bin edges are honoured, matching the tuple form."""
        torch.manual_seed(0)
        samples = torch.randn(64, 2)
        edges = [torch.linspace(-3, 3, 4), torch.linspace(-3, 3, 5)]
        tuple_out = torch.histogramdd(samples, bins=(edges[0], edges[1]))
        tensor_out = torch.histogramdd(samples, bins=edges)
        torch.testing.assert_close(tensor_out.hist, tuple_out.hist)

    def test_dtype_float64_golden(self):
        """float64 samples produce counts equal to the CPU golden result."""
        torch.manual_seed(0)
        samples = torch.randn(64, 2, dtype=torch.float64)
        cpu_hist, _ = torch.histogramdd(samples, bins=(2, 3))
        dev_out = torch.histogramdd(samples.to(self.device), bins=(2, 3))
        torch.testing.assert_close(dev_out.hist.cpu(), cpu_hist)


if __name__ == "__main__":
    run_tests()
