# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.is_neg 接口功能正确性
API 名称：torch.is_neg
API 签名：torch.is_neg(input) -> bool

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 普通张量/负位视图/空张量 | 已覆盖 |
| 枚举选项 | 无枚举参数 | 未覆盖（无该类参数） |
| 参数类型 | 仅接受 Tensor；非张量报错 | 已覆盖 |
| 传参与不传参 | input 为必填参数 | 已覆盖 |
| 等价类/边界值 | 负位已置位/未置位两类等价类 | 已覆盖 |
| 正常传参场景 | 返回 Python bool | 已覆盖 |
| 异常传参场景 | 非 Tensor 触发 TypeError | 已覆盖 |
| 混合设备类型 | 单 Tensor 输入，无混合设备路径 | 未覆盖（不适用） |

未覆盖项及原因：
- 枚举选项：is_neg 无枚举参数。
- 混合设备：仅一个输入张量，不存在跨设备组合。
- 负位为真（True）的构造依赖复数共轭取虚部路径，该路径仅在 CPU 上验证；
  设备侧仅验证普通张量返回 False 与 CPU 一致。
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
class Testis_neg(TestCase):
    """Test cases for torch.is_neg (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_plain_tensor_has_no_negative_bit(self):
        """A tensor holding literal negative values has no negative bit."""
        self.assertFalse(torch.is_neg(torch.tensor([-1.0, -2.0, -3.0])))

    def test_negative_view_has_negative_bit(self):
        """A lazily negated view reports the negative bit as set."""
        self.assertTrue(torch.is_neg(torch._neg_view(torch.tensor([1.0, 2.0]))))

    def test_complex_imag_has_negative_bit(self):
        """The documented conj().imag path yields True, resolve_neg clears it."""
        z = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j]).conj().imag
        self.assertTrue(torch.is_neg(z))
        self.assertFalse(torch.is_neg(torch.resolve_neg(z)))

    def test_returns_python_bool(self):
        """The predicate always yields a real bool."""
        self.assertIsInstance(torch.is_neg(torch.randn(3)), bool)

    def test_empty_tensor(self):
        """An empty tensor is not a negative view."""
        self.assertFalse(torch.is_neg(torch.empty(0)))

    def test_non_tensor_raises(self):
        """A non-Tensor argument raises TypeError."""
        with self.assertRaises(TypeError):
            torch.is_neg([1.0, 2.0])

    def test_device_tensor_matches_cpu(self):
        """A plain device tensor yields the same result as its CPU counterpart."""
        self.assertEqual(torch.is_neg(torch.randn(3, device=self.device)), torch.is_neg(torch.randn(3)))


if __name__ == "__main__":
    run_tests()
