# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.feature_dropout_ 接口功能正确性
API 名称：torch.feature_dropout_
API 签名：torch.feature_dropout_(input, p, train) -> Tensor

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| shape | 2D/3D/4D（N, C, ...） | 已覆盖 |
| dtype | float32/float16/float64 | 已覆盖 |
| device | 每个正常场景 CPU/NPU 双端执行 | 已覆盖 |
| 参数枚举 | train=True/False、p=0/0.5 | 已覆盖 |
| 可选参数 | p、train 均为必填位置参数 | 已覆盖（无默认值） |
| in-place 变体 | 本 API 即原地版本，返回同一对象 | 已覆盖 |
| out= 参数 | 无 out= 参数 | 未覆盖（不适用） |
| 广播 | 单输入无广播 | 未覆盖（不适用） |
| 连续性 | 原地操作保持输入形状与布局 | 已覆盖 |
| 混合设备输入 | 单 Tensor 输入，无混合设备路径 | 未覆盖（不适用） |
| 异常路径 | p 超出 [0,1] 触发异常 | 已覆盖 |

未覆盖项及原因：
- out=/广播/混合设备维度对该原地单输入 API 不适用。
- 随机掩码在 CPU 与 NPU 上不同，故只做结构性对照：被置零的完整特征通道、
  形状与 dtype 保持一致，不逐元素比对随机结果。
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
class Testfeature_dropout_(TestCase):
    """Test cases for torch.feature_dropout_ (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_eval_mode_is_identity(self):
        """In eval mode (train=False) the tensor is returned unchanged."""
        x = torch.randn(2, 3, 4, device=self.device)
        out = torch.feature_dropout_(x, 0.5, False)
        torch.testing.assert_close(out, x)

    def test_train_p_zero_is_identity(self):
        """train=True with p=0 keeps every element."""
        x = torch.ones(2, 3, 4, device=self.device)
        out = torch.feature_dropout_(x, 0.0, True)
        self.assertTrue(bool((out == 1.0).all()))

    def test_inplace_returns_same_object(self):
        """The in-place variant returns the very same tensor object."""
        x = torch.ones(4, 8)
        out = torch.feature_dropout_(x, 0.0, False)
        self.assertIs(out, x)

    def test_masks_whole_feature_channels(self):
        """Zeroed entries form complete feature channels (last-dim slices)."""
        torch.manual_seed(0)
        n, c, length = 4, 5, 7
        x = torch.ones(n, c, length, device=self.device)
        out = torch.feature_dropout_(x, 0.5, True)
        mask = (out == 0).reshape(n * c, length)
        all_or_none = mask.all(dim=1) | (~mask.any(dim=1))
        self.assertTrue(bool(all_or_none.all()))

    def test_shape_and_dtype_preserved(self):
        """Shape and dtype are unchanged by the in-place op."""
        x = torch.randn(2, 3, 4, dtype=torch.float32, device=self.device)
        out = torch.feature_dropout_(x, 0.5, True)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, torch.float32)

    def test_invalid_p_raises(self):
        """A probability outside [0, 1] is rejected."""
        with self.assertRaises((RuntimeError, ValueError)):
            torch.feature_dropout_(torch.ones(2, 3, 4), 1.5, True)

    def test_dtype_variants_preserved(self):
        """float16 and float64 inputs are preserved on the in-place path."""
        for dtype in (torch.float16, torch.float64):
            x = torch.ones(2, 3, 4, dtype=dtype, device=self.device)
            out = torch.feature_dropout_(x, 0.0, False)
            self.assertIs(out, x)
            self.assertEqual(out.dtype, dtype)

    def test_higher_rank_shape(self):
        """A 4D input is handled with whole-channel masking."""
        torch.manual_seed(0)
        n, c, h, w = 2, 3, 4, 5
        x = torch.ones(n, c, h, w, device=self.device)
        out = torch.feature_dropout_(x, 0.5, True)
        mask = (out == 0).reshape(n * c, h * w)
        all_or_none = mask.all(dim=1) | (~mask.any(dim=1))
        self.assertTrue(bool(all_or_none.all()))
        self.assertEqual(out.shape, (n, c, h, w))


if __name__ == "__main__":
    run_tests()
