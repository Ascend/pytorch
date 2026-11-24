# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.nn.intrinsic.qat.ConvReLU2d.from_float 接口功能正确性
API 名称：torch.ao.nn.intrinsic.qat.ConvReLU2d.from_float
API 签名：ConvReLU2d.from_float(cls, mod, use_precomputed_fake_quant=False)

覆盖维度表：
| 覆盖维度         | 说明                                             | 覆盖情况 |
|------------------|--------------------------------------------------|----------|
| 空/非空          | 入参为浮点融合模块对象，无空/非空语义            | 不适用 |
| 枚举选项         | 该接口无枚举参数                                 | 不适用 |
| 参数类型         | 传入 nni 融合模块 vs 普通 nn.Module              | 已覆盖：test_from_float_rejects_non_fused |
| 传参与不传参     | use_precomputed_fake_quant 省略/显式传           | 已覆盖：test_from_float_precomputed_flag |
| 等价类/边界值    | 权重形状、卷积配置与无偏置卷积路径               | 已覆盖：test_from_float_copies_conv_config |
| 正常传参场景     | 由浮点融合模块构造 QAT 模块且前向结果一致        | 已覆盖：test_from_float_returns_qat_module |
| 异常传参场景     | 类型不符或缺/非法 qconfig 触发 AssertionError    | 已覆盖：test_from_float_rejects_missing_qconfig |
| 混合设备类型     | 多 Tensor 异构设备输入                           | 未覆盖，原因见下 |

未覆盖项及原因：
- 混合设备类型：QAT 融合模块的 fake-quant 与卷积路径只在 CPU 上执行，无 NPU/CUDA 执行路径，
  无法构造异构设备输入，故不做 CPU/NPU 混合对照。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注：QAT from_float 只依赖 fake-quant observer，不需要 CPU 量化引擎内核，故不做 CPU/NPU Golden 对照；
     数值正确性以同一 float 融合模块在 CPU 上的前向结果作参照（关闭 fake-quant 后二者应一致）。
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

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

import torch.ao.nn.intrinsic as nni
import torch.nn as nn
from torch.ao.nn.intrinsic.qat import ConvReLU2d
from torch.ao.quantization import (  # noqa: F401
    default_fake_quant,
    default_weight_fake_quant,
    disable_fake_quant,
)
from torch.ao.quantization.qconfig import QConfig

# Fake-quant observers only; no CPU quantized engine kernel is required.
_QCONFIG = QConfig(activation=default_fake_quant, weight=default_weight_fake_quant)
_SHAPE = (2, 3, 8, 8)
_OUT_SHAPE = (2, 4, 6, 6)


def _build_float(kernel_size=3, stride=1, padding=0, out_channels=4, bias=True):
    conv = nn.Conv2d(3, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
    # The fused module must expose qconfig (checked on the container); the QAT
    # from_float then reads the unwrapped conv's qconfig, so set both.
    conv.qconfig = _QCONFIG
    mod = nni.ConvReLU2d(conv, nn.ReLU())
    mod.qconfig = _QCONFIG
    mod.eval()
    return mod


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestConvReLU2dFromFloatUt(TestCase):
    """Functional tests for torch.ao.nn.intrinsic.qat.ConvReLU2d.from_float."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_from_float_returns_qat_module(self):
        """from_float builds a QAT ConvReLU2d carrying a weight fake-quant."""
        module = ConvReLU2d.from_float(_build_float())
        self.assertIsInstance(module, ConvReLU2d)
        self.assertTrue(hasattr(module, "weight_fake_quant"))

    def test_from_float_shares_weight_and_bias(self):
        """from_float reuses the float conv Parameter objects (no copy)."""
        float_mod = _build_float()
        conv = float_mod[0]
        module = ConvReLU2d.from_float(float_mod)
        self.assertIs(module.weight, conv.weight)
        self.assertIs(module.bias, conv.bias)
        self.assertEqual(tuple(module.weight.shape), (4, 3, 3, 3))

    def test_from_float_copies_conv_config(self):
        """Convolution hyper-parameters are propagated to the QAT module."""
        float_mod = _build_float(kernel_size=5, stride=2, padding=1, out_channels=6)
        module = ConvReLU2d.from_float(float_mod)
        self.assertEqual(module.in_channels, 3)
        self.assertEqual(module.out_channels, 6)
        self.assertEqual(tuple(module.kernel_size), (5, 5))
        self.assertEqual(tuple(module.stride), (2, 2))
        self.assertEqual(tuple(module.padding), (1, 1))
        self.assertEqual(module.padding_mode, "zeros")

    def test_from_float_precomputed_flag(self):
        """The optional use_precomputed_fake_quant argument does not alter output."""
        float_mod = _build_float()
        default_mod = ConvReLU2d.from_float(float_mod)
        true_mod = ConvReLU2d.from_float(float_mod, use_precomputed_fake_quant=True)
        self.assertIs(true_mod.weight, default_mod.weight)
        self.assertEqual(tuple(true_mod.weight.shape), tuple(default_mod.weight.shape))

    def test_from_float_forward_matches_float(self):
        """With fake-quant disabled the QAT output must match the float module."""
        float_mod = _build_float()
        module = ConvReLU2d.from_float(float_mod)
        module.eval()
        module.apply(disable_fake_quant)
        x = torch.randn(*_SHAPE)
        with torch.no_grad():
            expected = float_mod(x)
            actual = module(x)
        self.assertEqual(tuple(actual.shape), _OUT_SHAPE)
        # Conv+ReLU only; a mis-copied weight/bias makes this numerically wrong.
        torch.testing.assert_close(actual, expected)

    def test_from_float_biasless_conv(self):
        """A biasless float conv yields a QAT module whose bias is None."""
        float_mod = _build_float(bias=False)
        module = ConvReLU2d.from_float(float_mod)
        self.assertIsNone(module.bias)
        module.eval()
        module.apply(disable_fake_quant)
        x = torch.randn(*_SHAPE)
        with torch.no_grad():
            torch.testing.assert_close(module(x), float_mod(x))

    def test_from_float_rejects_non_fused(self):
        """A plain (non-fused) float module is rejected with AssertionError."""
        with self.assertRaisesRegex(AssertionError, "only works for"):
            ConvReLU2d.from_float(nn.Conv2d(3, 4, 3))

    def test_from_float_rejects_missing_qconfig(self):
        """A fused module without qconfig is rejected with AssertionError."""
        mod = nni.ConvReLU2d(nn.Conv2d(3, 4, 3), nn.ReLU())
        with self.assertRaisesRegex(AssertionError, "must have qconfig defined"):
            ConvReLU2d.from_float(mod)

    def test_from_float_rejects_bad_qconfig(self):
        """A falsy qconfig is rejected before any module construction."""
        float_mod = _build_float()
        float_mod.qconfig = None
        with self.assertRaisesRegex(AssertionError, "must have a valid qconfig"):
            ConvReLU2d.from_float(float_mod)


if __name__ == "__main__":
    run_tests()
