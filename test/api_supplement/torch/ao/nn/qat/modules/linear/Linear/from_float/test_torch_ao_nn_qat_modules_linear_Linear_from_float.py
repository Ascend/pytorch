# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.nn.qat.modules.linear.Linear.from_float 接口功能正确性
API 名称：torch.ao.nn.qat.modules.linear.Linear.from_float
API 签名：Linear.from_float(cls, mod, use_precomputed_fake_quant=False)

覆盖维度表：
| 覆盖维度      | 说明                                   | 覆盖情况                                                |
|---------------|----------------------------------------|---------------------------------------------------------|
| 空/非空       | mod 无 qconfig（空）与带合法           | 已覆盖：test_from_float_requires_qconfig /              |
|               | qconfig（非空）                        | test_from_float_returns_qat_linear                      |
| 枚举选项      | 无枚举参数；bias 取 True/False 两态    | 已覆盖：test_from_float_shares_weight_and_bias /        |
|               |                                        | test_from_float_no_bias                                 |
| 参数类型      | mod 须为 torch.nn.Linear 实例          | 已覆盖：test_from_float_rejects_non_linear              |
| 传参与不传参  | 省略 use_precomputed_fake_quant        | 已覆盖：test_from_float_returns_qat_linear /            |
|               | 用默认 False；显式传 True              | test_from_float_use_precomputed_flag                    |
| 等价类/边界值 | 典型 (4,5) 维度与 no-bias 变体         | 已覆盖：test_from_float_returns_qat_linear /            |
|               |                                        | test_from_float_no_bias                                 |
| 正常传参场景  | 由 float Linear 生成 QAT Linear        | 已覆盖：test_from_float_forward_shape /                 |
|               | 且共享权重/偏置                        | test_from_float_shares_weight_and_bias                  |
| 异常传参场景  | 非 Linear、缺 qconfig、qconfig=None    | 已覆盖：三个 test_from_float_rejects_/requires_ 方法    |
|               | 各抛 AssertionError                    |                                                         |
| 混合设备类型  | 单模块转换接口，不涉及多 Tensor        | 不适用                                                  |
|               | 异构设备输入                           |                                                         |

未覆盖项及原因：
- 混合设备类型：QAT Linear 的 fake-quant 权重仅由 CPU 量化后端提供，无 NPU/CUDA 执行路径，
  无法构造异构设备输入。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注：QAT Linear 只在 CPU 量化后端上有 fake-quant 语义，无 NPU/CUDA 数值对照路径，
     故不做 CPU/NPU Golden 对照，仅验证转换正确性与权重/偏置共享语义。
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

import torch.ao.quantization
import torch.nn
from torch.ao.nn.qat.modules.linear import Linear as QATLinear

_QCONFIG = torch.ao.quantization.default_qat_qconfig


def _make_float_linear(bias=True):
    mod = torch.nn.Linear(4, 5, bias=bias)
    mod.qconfig = _QCONFIG
    return mod


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestTorchAoNnQatModulesLinearFromFloatUt(TestCase):
    """Functional tests for torch.ao.nn.qat.modules.linear.Linear.from_float."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_from_float_returns_qat_linear(self):
        """A float Linear is converted into a QAT Linear with matching feature dims."""
        module = QATLinear.from_float(_make_float_linear())
        self.assertIsInstance(module, QATLinear)
        self.assertEqual(module.in_features, 4)
        self.assertEqual(module.out_features, 5)
        self.assertIs(module.qconfig, _QCONFIG)

    def test_from_float_shares_weight_and_bias(self):
        """from_float reuses the exact weight and bias objects of the float module."""
        float_mod = _make_float_linear()
        module = QATLinear.from_float(float_mod)
        # A wrong implementation that deep copies the params would fail these identity checks.
        self.assertIs(module.weight, float_mod.weight)
        self.assertIs(module.bias, float_mod.bias)

    def test_from_float_no_bias(self):
        """A bias-free float Linear yields a bias-free QAT Linear."""
        module = QATLinear.from_float(_make_float_linear(bias=False))
        self.assertIsNone(module.bias)

    def test_from_float_forward_shape(self):
        """The converted module forwards to the expected output shape and dtype."""
        module = QATLinear.from_float(_make_float_linear())
        out = module(torch.randn(3, 4))
        self.assertEqual(tuple(out.shape), (3, 5))
        self.assertEqual(out.dtype, torch.float32)

    def test_from_float_use_precomputed_flag(self):
        """Passing use_precomputed_fake_quant=True is accepted and still converts."""
        module = QATLinear.from_float(_make_float_linear(), use_precomputed_fake_quant=True)
        self.assertIsInstance(module, QATLinear)

    def test_from_float_rejects_non_linear(self):
        """A non-Linear float module is rejected."""
        other = torch.nn.Conv2d(4, 5, 3)
        other.qconfig = _QCONFIG
        with self.assertRaisesRegex(AssertionError, "only works for Linear"):
            QATLinear.from_float(other)

    def test_from_float_requires_qconfig(self):
        """A float module without a qconfig attribute is rejected."""
        with self.assertRaisesRegex(AssertionError, "must have qconfig defined"):
            QATLinear.from_float(torch.nn.Linear(4, 5))

    def test_from_float_rejects_invalid_qconfig(self):
        """A float module with qconfig=None is rejected."""
        float_mod = torch.nn.Linear(4, 5)
        float_mod.qconfig = None
        with self.assertRaisesRegex(AssertionError, "must have a valid qconfig"):
            QATLinear.from_float(float_mod)


if __name__ == "__main__":
    run_tests()
