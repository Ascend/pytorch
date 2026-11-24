# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.nn.quantized.dynamic.modules.linear.Linear.from_float 接口功能正确性
API 名称：torch.ao.nn.quantized.dynamic.modules.linear.Linear.from_float
API 签名：Linear.from_float(cls, mod, use_precomputed_fake_quant=False)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | 偏置为空(bias=False)与非空 | 已覆盖：test_from_float_without_bias、test_from_float_preserves_bias |
| 枚举选项 | 权重 dtype 枚举 qint8/float16 | 已覆盖：test_from_float_returns_qint8_linear；float16 未覆盖，原因见下 |
| 参数类型 | 接受 nn.Linear/NonDynamicallyQuantizableLinear/LinearReLU | 已覆盖：test_from_float_accepts_supported_float_types |
| 传参与不传参 | use_precomputed_fake_quant 省略与显式传 False | 已覆盖：test_from_float_default_use_precomputed_flag |
| 等价类/边界值 | in_features/out_features 复制一致 | 已覆盖：test_from_float_copies_features |
| 正常传参场景 | 量化前向输出 shape/dtype 及与 op 结果一致 | 已覆盖：test_from_float_matches_quantized_op |
| 异常传参场景 | 非法模块或缺 qconfig 触发 AssertionError | 已覆盖：test_from_float_rejects_non_linear、test_from_float_requires_qconfig |
| 混合设备类型 | 多 Tensor 异构设备输入 | 未覆盖，原因见下 |

未覆盖项及原因：
- 枚举选项(float16)：float16 动态线性内核仅由 fbgemm/onednn 后端提供，qnnpack 平台无 fp16 内核，
  故本用例仅在 qint8 路径上验证；float16 路径在无对应内核的机器上不可构造。
- 混合设备类型：动态量化模块仅由 CPU 量化后端（onednn/fbgemm/qnnpack）提供内核，
  NPU/CUDA 上无对应执行路径，无法构造异构设备输入。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注：动态量化 Linear 只在 CPU 量化后端执行，无 NPU/CUDA 数值对照路径，
     故本用例不做 CPU/NPU Golden 对照，仅验证量化后端上的功能语义。
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

import torch.ao.nn.intrinsic
import torch.ao.nn.quantized.dynamic
from torch.ao.nn.quantized.dynamic.modules.linear import Linear

_ENGINE_CANDIDATES = ("onednn", "fbgemm", "qnnpack")
_IN_FEATURES = 4
_OUT_FEATURES = 3


def _use_quantized_engine(probe=None):
    """Select a CPU quantized engine able to run the target op.

    The dynamic quantized Linear only provides kernels on the CPU quantized
    backend, so there is no NPU/CUDA execution path to compare against.
    """
    supported = torch.backends.quantized.supported_engines
    for name in _ENGINE_CANDIDATES:
        if name not in supported:
            continue
        torch.backends.quantized.engine = name
        if probe is None:
            return name
        try:
            probe()
        except RuntimeError:
            # This engine has no kernel for the target op; try the next one.
            continue
        return name
    raise unittest.SkipTest("No CPU quantized engine can run the target op on this platform")


def _build_float_linear(bias=True, qconfig=None):
    if qconfig is None:
        qconfig = torch.ao.quantization.default_dynamic_qconfig
    mod = torch.nn.Linear(_IN_FEATURES, _OUT_FEATURES, bias=bias)
    mod.qconfig = qconfig
    return mod


def _build_non_dynamically_quantizable_linear():
    mod = torch.nn.modules.linear.NonDynamicallyQuantizableLinear(_IN_FEATURES, _OUT_FEATURES)
    mod.qconfig = torch.ao.quantization.default_dynamic_qconfig
    return mod


def _build_linear_relu():
    inner = torch.nn.Linear(_IN_FEATURES, _OUT_FEATURES)
    inner.qconfig = torch.ao.quantization.default_dynamic_qconfig
    fused = torch.ao.nn.intrinsic.LinearReLU(inner, torch.nn.ReLU())
    fused.qconfig = torch.ao.quantization.default_dynamic_qconfig
    return fused


def _probe():
    module = Linear.from_float(_build_float_linear())
    module(torch.randn(2, _IN_FEATURES))


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestLinearFromfloatUt(TestCase):
    """Functional tests for the dynamic quantized Linear.from_float classmethod."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        _use_quantized_engine(_probe)

    def test_from_float_returns_qint8_linear(self):
        module = Linear.from_float(_build_float_linear())
        self.assertIsInstance(module, Linear)
        self.assertEqual(module._packed_params.dtype, torch.qint8)

    def test_from_float_copies_features(self):
        module = Linear.from_float(_build_float_linear())
        self.assertEqual(module.in_features, _IN_FEATURES)
        self.assertEqual(module.out_features, _OUT_FEATURES)

    def test_from_float_preserves_bias(self):
        float_mod = _build_float_linear()
        module = Linear.from_float(float_mod)
        self.assertIsNotNone(module.bias())
        self.assertEqual(tuple(module.bias().shape), (_OUT_FEATURES,))

    def test_from_float_without_bias(self):
        module = Linear.from_float(_build_float_linear(bias=False))
        out = module(torch.randn(2, _IN_FEATURES))
        self.assertEqual(tuple(out.shape), (2, _OUT_FEATURES))
        self.assertEqual(out.dtype, torch.float32)

    def test_from_float_matches_quantized_op(self):
        module = Linear.from_float(_build_float_linear())
        x = torch.randn(5, _IN_FEATURES)
        expected = torch.ops.quantized.linear_dynamic(
            x, module._packed_params._packed_params, reduce_range=True
        )
        out = module(x)
        self.assertEqual(out.dtype, torch.float32)
        torch.testing.assert_close(out, expected)

    def test_from_float_accepts_supported_float_types(self):
        for mod in (
            _build_float_linear(),
            _build_non_dynamically_quantizable_linear(),
            _build_linear_relu(),
        ):
            self.assertIsInstance(Linear.from_float(mod), Linear)

    def test_from_float_default_use_precomputed_flag(self):
        default_mod = Linear.from_float(_build_float_linear())
        explicit_mod = Linear.from_float(_build_float_linear(), use_precomputed_fake_quant=False)
        self.assertIsInstance(default_mod, Linear)
        self.assertIsInstance(explicit_mod, Linear)
        self.assertEqual(default_mod.in_features, explicit_mod.in_features)
        self.assertEqual(default_mod.out_features, explicit_mod.out_features)

    def test_from_float_rejects_non_linear(self):
        with self.assertRaises(AssertionError):
            Linear.from_float(torch.nn.ReLU())

    def test_from_float_requires_qconfig(self):
        with self.assertRaises(AssertionError):
            Linear.from_float(torch.nn.Linear(_IN_FEATURES, _OUT_FEATURES))


if __name__ == "__main__":
    run_tests()
