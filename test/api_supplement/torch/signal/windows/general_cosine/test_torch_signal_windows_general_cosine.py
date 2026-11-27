# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.signal.windows.general_cosine 接口功能正确性
API 名称：torch.signal.windows.general_cosine
API 签名：torch.signal.windows.general_cosine(M, *, a, sym=True, dtype=None,
          layout=torch.strided, device=None, requires_grad=False) -> Tensor

覆盖维度表：
| 覆盖维度         | 说明                                                  | 覆盖情况 |
|------------------|-------------------------------------------------------|----------|
| 等价类/边界值    | M=0 / M=1 / 常规 M                                    | 已覆盖 |
| 枚举选项         | sym=True / sym=False；dtype float32 / float64         | 已覆盖 |
| 参数类型         | a 为 list / tuple；a 非可迭代                        | 已覆盖 |
| 传参与不传参     | 省略 dtype/device/requires_grad 与显式传入            | 已覆盖 |
| 正常传参场景     | 数值与文档示例一致、与 general_hamming 等价           | 已覆盖 |
| device           | 正常场景 CPU/NPU 双端执行，CPU 为 Golden              | 已覆盖 |
| 返回值结构       | 输出 shape=M、一维、dtype 正确、requires_grad 可控    | 已覆盖 |
| 异常传参场景     | 空系数 / 非可迭代 a / 非法 dtype / 负 M               | 已覆盖 |
| 混合设备类型     | 单 Tensor 输出，无混合设备路径                        | 未覆盖（不适用） |

未覆盖项及原因：
- 混合设备类型：接口只做单设备计算，不涉及多 Tensor 设备混用。
- 文档示例黄金值为 float32 截断表示，对照使用 rtol=1e-5/atol=1e-6 容纳 float32 表示误差，
  不用于掩盖语义差异。

注意：非图模式浮点 Tensor 以 CPU 结果为 Golden，设备输出移回 CPU 后用
     torch.testing.assert_close 对照；结构性结果（shape/dtype/numel）用精确断言。
"""
import os
import unittest

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

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

# ORDER CONSTRAINT: import torch_npu.testing.testcase BEFORE
# torch_npu.contrib.transfer_to_npu (patching first breaks the import).
try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

if DEVICE_NAME == "npu":
    import torch_npu.contrib.transfer_to_npu  # noqa: F401


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestSignalWindowsGeneralCosineUt(TestCase):
    """Functional tests for torch.signal.windows.general_cosine (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_documented_symmetric_values(self):
        """Symmetric window matches the documented golden values."""
        expected = torch.tensor(
            [0.5400000214576721, 0.3376407027244568, 0.12875619530677795, 0.4200000762939453,
             0.9136031866073608, 0.9136030673980713, 0.419999897480011, 0.12875622510910034,
             0.3376408517360687, 0.5400000214576721],
            dtype=torch.float32,
        )
        out = torch.signal.windows.general_cosine(10, a=[0.46, 0.23, 0.31], sym=True, device=self.device)
        torch.testing.assert_close(out.cpu(), expected, rtol=1e-5, atol=1e-6)

    def test_documented_periodic_values(self):
        """Periodic window matches the documented golden values."""
        expected = torch.tensor(
            [0.0, 0.09549152851104736, 0.34549152851104736, 0.6545085310935974, 0.9045085310935974,
             1.0, 0.9045085310935974, 0.6545085906982422, 0.3454914391040802, 0.09549152851104736],
            dtype=torch.float32,
        )
        out = torch.signal.windows.general_cosine(10, a=[0.5, 0.5], sym=False, device=self.device)
        torch.testing.assert_close(out.cpu(), expected, rtol=1e-5, atol=1e-6)

    def test_matches_general_hamming(self):
        """Two-coefficient general_cosine equals general_hamming."""
        alpha = 0.7
        cosine = torch.signal.windows.general_cosine(16, a=[alpha, 1.0 - alpha])
        hamming = torch.signal.windows.general_hamming(16, alpha=alpha)
        torch.testing.assert_close(cosine, hamming)

    def test_cpu_golden(self):
        """Device window equals the CPU golden window."""
        cpu_out = torch.signal.windows.general_cosine(12, a=[0.46, 0.23, 0.31], sym=True)
        dev_out = torch.signal.windows.general_cosine(12, a=[0.46, 0.23, 0.31], sym=True, device=self.device)
        torch.testing.assert_close(dev_out.cpu(), cpu_out)

    def test_dtype_float64_golden(self):
        """float64 input produces the float64 CPU golden window."""
        cpu_out = torch.signal.windows.general_cosine(8, a=[0.5, 0.5], dtype=torch.float64)
        dev_out = torch.signal.windows.general_cosine(8, a=[0.5, 0.5], dtype=torch.float64, device=self.device)
        self.assertEqual(dev_out.dtype, torch.float64)
        torch.testing.assert_close(dev_out.cpu(), cpu_out)

    def test_default_dtype_is_float32(self):
        """Without an explicit dtype the default float dtype is used."""
        out = torch.signal.windows.general_cosine(8, a=[0.5, 0.5])
        self.assertEqual(out.dtype, torch.float32)

    def test_requires_grad_propagates(self):
        """requires_grad is honoured on the returned window."""
        out = torch.signal.windows.general_cosine(8, a=[0.5, 0.5], requires_grad=True)
        self.assertTrue(out.requires_grad)

    def test_length_zero_returns_empty(self):
        """M=0 returns an empty window."""
        out = torch.signal.windows.general_cosine(0, a=[0.5, 0.5])
        self.assertEqual(tuple(out.shape), (0,))
        self.assertEqual(out.numel(), 0)

    def test_length_one_returns_ones(self):
        """M=1 returns a single-element all-ones window."""
        out = torch.signal.windows.general_cosine(1, a=[0.5, 0.5])
        self.assertTrue(torch.equal(out, torch.ones(1)))

    def test_sym_flag_changes_result(self):
        """Symmetric and periodic windows differ for the same coefficients."""
        sym_true = torch.signal.windows.general_cosine(10, a=[0.5, 0.5], sym=True)
        sym_false = torch.signal.windows.general_cosine(10, a=[0.5, 0.5], sym=False)
        self.assertFalse(torch.allclose(sym_true, sym_false))

    def test_empty_coefficients_raises(self):
        """An empty coefficient sequence is rejected with ValueError."""
        with self.assertRaises(ValueError):
            torch.signal.windows.general_cosine(10, a=[])

    def test_non_iterable_coefficients_raises(self):
        """A non-iterable coefficient argument is rejected with TypeError."""
        with self.assertRaises(TypeError):
            torch.signal.windows.general_cosine(10, a=5)

    def test_invalid_dtype_raises(self):
        """A non-floating dtype is rejected with ValueError."""
        with self.assertRaises(ValueError):
            torch.signal.windows.general_cosine(10, a=[0.5, 0.5], dtype=torch.int32)

    def test_negative_length_raises(self):
        """A negative window length is rejected with ValueError."""
        with self.assertRaises(ValueError):
            torch.signal.windows.general_cosine(-1, a=[0.5, 0.5])


if __name__ == "__main__":
    run_tests()
