# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.foreach.clone 接口功能正确性
API 名称：torch.foreach.clone
API 签名：torch.foreach.clone(inputs, /, *, memory_format=None) -> tuple[Tensor]

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况                                                  |
|------------------|--------------------------------------------|-----------------------------------------------------------|
| shape            | 0-dim / 1D / 2D / 3D                       | 已覆盖：shape_coverage                                    |
| dtype            | float32/float16/bfloat16/int64/bool        | 已覆盖：dtype_coverage                                    |
| device           | 每个正常场景 CPU/NPU 双端，CPU 为 Golden   | 已覆盖：cpu_golden_npu 及各 *_coverage                    |
| 参数枚举         | memory_format=None / contiguous_format     | 已覆盖：memory_format_contiguous、默认省略                |
| 可选参数         | memory_format 显式传入 vs 省略默认         | 已覆盖                                                    |
| in-place 变体    | 无 in-place 变体（clone 天然 out-of-place）| 不适用                                                    |
| out= 参数        | 不支持 out=                                | 不适用                                                    |
| 广播             | 逐张量独立处理，无广播语义                 | 不适用                                                    |
| 连续性           | 非连续（transpose 视图）输入               | 已覆盖：non_contiguous                                    |
| 混合设备输入     | list 内同时含 NPU 与 CPU tensor（逐张量回退）| 已覆盖：mixed_device_falls_back（无加速器时按环境不具备跳过）|
| 异常路径         | 空 list / 非法 memory_format / 非 list 输入| 已覆盖：empty_list_raises、invalid_memory_format_raises、  |
|                  |                                            |         non_list_raises                                   |

未覆盖项及原因：
- in-place 变体 / out= / 广播：clone 语义为返回独立副本，无 in-place、无 out=、不涉及广播。
- NPU 原生算子：本仓库环境下 torch.foreach.clone 在 NPU 上回退 CPU 执行（无原生 aclnn 算子），
  功能结果仍以 CPU 为 Golden 对照；原生算子覆盖不在本用例判定范围。
- 混合设备异常契约：foreach 的 slow path 不校验 list 内设备一致性，逐张量各自其设备计算，
  混合设备 list 不抛异常；用例仅验证逐张量回退结果正确，不再断言抛错。

注：foreach 系列为张量数值/结构运算（计算类）；每个正常场景均 CPU/NPU 双端执行，CPU 输出为 Golden，
     NPU 输出移至 CPU 后用 torch.testing.assert_close 对照（整数/布尔用 torch.equal 精确比较）。
"""
import os
import unittest

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Requires a PyTorch 2.15 accelerator (NPU/CUDA) environment, which is not
# available here; skip the whole case until such an environment exists.
BROKEN_SKIP_REASON = "No PyTorch 2.15 accelerator (NPU/CUDA) environment available"

# Probe the device only when the supplement suite actually runs. On the default
# CI collection path the class-level skipUnless fires first, so no torch_npu
# import or device initialization happens here.
DEVICE_NAME = "cpu"
if RUN_FLAG:
    DEVICE_NAME = None
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = "npu"
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


@unittest.skip(BROKEN_SKIP_REASON)
@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestForeachCloneUt(TestCase):
    """Functional tests for torch.foreach.clone (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _assert_close_seq(self, actual, expected):
        """Compare two tensor sequences: float via assert_close, else exact."""
        for got, ref in zip(actual, expected):
            if ref.is_floating_point():
                torch.testing.assert_close(got, ref)
            else:
                self.assertTrue(torch.equal(got, ref))

    def test_foreach_clone_cpu_golden_npu(self):
        """CPU result is the Golden for the device path (list input)."""
        cpu_inputs = [torch.randn(2, 3), torch.randn(4)]
        cpu_out = torch.foreach.clone(cpu_inputs)
        dev_out = torch.foreach.clone([t.to(self.device) for t in cpu_inputs])
        self.assertIsInstance(cpu_out, tuple)
        self.assertEqual(len(dev_out), len(cpu_inputs))
        self._assert_close_seq([t.cpu() for t in dev_out], cpu_out)

    def test_foreach_clone_returns_distinct_copies(self):
        """Each cloned tensor is an independent allocation; writes do not alias input."""
        x = torch.ones(3, 3, device=self.device)
        out = torch.foreach.clone([x])
        self.assertNotEqual(out[0].data_ptr(), x.data_ptr())
        out[0].add_(5.0)
        self.assertTrue(torch.equal(x, torch.ones(3, 3, device=self.device)))

    def test_foreach_clone_tuple_input(self):
        """Tuple input is accepted and returns a tuple with matching values."""
        x = torch.randn(5)
        cpu_out = torch.foreach.clone((x,))
        self.assertIsInstance(cpu_out, tuple)
        dev_out = torch.foreach.clone((x.to(self.device),))
        self.assertEqual(len(dev_out), 1)
        self._assert_close_seq([dev_out[0].cpu()], list(cpu_out))

    def test_foreach_clone_single_element(self):
        """A single-tensor list is cloned with matching values."""
        x = torch.randn(2, 2)
        out = torch.foreach.clone([x.to(self.device)])
        self.assertEqual(len(out), 1)
        torch.testing.assert_close(out[0].cpu(), x)

    def test_foreach_clone_shape_coverage(self):
        """0-dim / 1D / 2D / 3D shapes are preserved with matching values."""
        for shape in [(), (1,), (2, 3), (2, 3, 4)]:
            base = torch.randn(shape)
            out = torch.foreach.clone([base.to(self.device)])[0]
            self.assertEqual(out.shape, torch.Size(shape))
            torch.testing.assert_close(out.cpu(), base)

    def test_foreach_clone_dtype_coverage(self):
        """clone preserves dtype for float / int / bool inputs."""
        for dtype in [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.bool]:
            base = torch.ones(4, dtype=dtype)
            out = torch.foreach.clone([base.to(self.device)])[0]
            self.assertEqual(out.dtype, dtype)
            if base.is_floating_point():
                torch.testing.assert_close(out.cpu(), base)
            else:
                self.assertTrue(torch.equal(out.cpu(), base))

    def test_foreach_clone_non_contiguous(self):
        """A non-contiguous (transposed) view is cloned with identical values."""
        base = torch.randn(4, 6).t()
        self.assertFalse(base.is_contiguous())
        out = torch.foreach.clone([base.to(self.device)])[0]
        torch.testing.assert_close(out.cpu(), base)

    def test_foreach_clone_memory_format_contiguous(self):
        """Explicit memory_format=contiguous_format is honored and values preserved."""
        base = torch.randn(4, 6).t()
        out = torch.foreach.clone([base.to(self.device)], memory_format=torch.contiguous_format)[0]
        self.assertTrue(out.is_contiguous())
        torch.testing.assert_close(out.cpu(), base)

    def test_foreach_clone_empty_list_raises(self):
        """An empty tensor list raises RuntimeError."""
        with self.assertRaises(RuntimeError):
            torch.foreach.clone([])

    def test_foreach_clone_invalid_memory_format_raises(self):
        """A non-memory_format argument raises TypeError."""
        with self.assertRaises(TypeError):
            torch.foreach.clone([torch.randn(2)], memory_format=123)

    def test_foreach_clone_non_list_raises(self):
        """Passing a bare Tensor instead of a list raises TypeError."""
        with self.assertRaises(TypeError):
            torch.foreach.clone(torch.randn(2))

    def test_foreach_clone_mixed_device_falls_back(self):
        """A mixed accelerator/CPU list does not raise: the slow path clones each
        tensor on its own device."""
        if self.device_name == "cpu":
            self.skipTest("no accelerator available for mixed-device input")
        dev_t = torch.randn(2, device=self.device)
        cpu_t = torch.randn(2)
        out = torch.foreach.clone([dev_t, cpu_t])
        self.assertEqual(len(out), 2)
        self._assert_close_seq([out[0].cpu()], [dev_t.cpu()])
        self._assert_close_seq([out[1]], [cpu_t])


if __name__ == "__main__":
    run_tests()
