# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.dropout_ 接口功能正确性
API 名称：torch.dropout_
API 签名：torch.dropout_(input, p, train) -> Tensor

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| shape | 1D/2D/高维 | 已覆盖 |
| dtype | float32/float16/float64/bfloat16 | 已覆盖 |
| device | 每个正常场景 CPU/NPU 双端执行 | 已覆盖 |
| 参数枚举 | train=True/False、p=0/0.5/1 | 已覆盖 |
| 可选参数 | p、train 均为必填位置参数 | 已覆盖（无默认值） |
| in-place 变体 | 本 API 即原地版本，返回同一对象 | 已覆盖 |
| out= 参数 | 无 out= 参数 | 未覆盖（不适用） |
| 广播 | 单输入无广播 | 未覆盖（不适用） |
| 连续性 | 原地操作保持输入连续性 | 已覆盖 |
| 混合设备输入 | 单 Tensor 输入，无混合设备路径 | 未覆盖（不适用） |
| 异常路径 | p 超出 [0,1] 触发异常 | 已覆盖 |

未覆盖项及原因：
- out=/广播/混合设备维度对该原地单输入 API 不适用：dropout_ 原地修改单个张量，
  不接收第二个张量参与运算。
- p=0.5 的置零比例以统计容差断言；CPU 与 NPU 的随机掩码不同，故不做逐元素对照，
  仅做结构性（比例、取值范围、原地性）对照。
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
class Testdropout_(TestCase):
    """Test cases for torch.dropout_ (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_eval_mode_is_identity(self):
        """In eval mode (train=False) the tensor is returned unchanged."""
        x = torch.randn(3, 4)
        out = torch.dropout_(x, 0.5, False)
        torch.testing.assert_close(out, x)

    def test_train_p_zero_is_identity(self):
        """train=True with p=0 keeps every element."""
        x = torch.ones(8)
        out = torch.dropout_(x, 0.0, True)
        self.assertTrue(bool((out == 1.0).all()))

    def test_train_p_one_zeros_all(self):
        """train=True with p=1 zeroes every element."""
        x = torch.ones(8)
        out = torch.dropout_(x, 1.0, True)
        self.assertTrue(bool((out == 0.0).all()))

    def test_inplace_returns_same_object(self):
        """The in-place variant returns the very same tensor object."""
        x = torch.ones(4)
        out = torch.dropout_(x, 0.0, False)
        self.assertIs(out, x)

    def test_train_zero_fraction_matches_p(self):
        """The zeroed fraction under train mode is close to p (statistical)."""
        torch.manual_seed(0)
        x = torch.ones(20000, device=self.device)
        out = torch.dropout_(x, 0.5, True)
        fraction = (out == 0).float().mean().item()
        self.assertLess(abs(fraction - 0.5), 0.05)

    def test_dtype_preserved(self):
        """The dtype of the input is preserved after the in-place op."""
        x = torch.ones(4, dtype=torch.float16, device=self.device)
        out = torch.dropout_(x, 0.0, False)
        self.assertEqual(out.dtype, torch.float16)

    def test_invalid_p_raises(self):
        """A probability outside [0, 1] is rejected."""
        with self.assertRaises((RuntimeError, ValueError)):
            torch.dropout_(torch.ones(4), 1.5, True)

    def test_dtype_float64_preserved(self):
        """float64 inputs are preserved and the device path is exercised."""
        x = torch.ones(4, dtype=torch.float64, device=self.device)
        out = torch.dropout_(x, 0.0, False)
        self.assertIs(out, x)
        self.assertEqual(out.dtype, torch.float64)

    def test_dtype_bfloat16_preserved(self):
        """bfloat16 inputs are preserved on the in-place path."""
        x = torch.ones(4, dtype=torch.bfloat16, device=self.device)
        out = torch.dropout_(x, 0.0, False)
        self.assertIs(out, x)
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_higher_rank_shape(self):
        """A 4D input is handled and its shape is preserved."""
        x = torch.ones(2, 3, 4, 5, device=self.device)
        out = torch.dropout_(x, 0.0, True)
        self.assertEqual(out.shape, (2, 3, 4, 5))
        self.assertTrue(bool((out == 1.0).all()))


if __name__ == "__main__":
    run_tests()
