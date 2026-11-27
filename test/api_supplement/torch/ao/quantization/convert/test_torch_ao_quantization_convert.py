# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.convert 接口功能正确性
API 名称：torch.ao.quantization.convert
API 签名：convert(module, mapping=None, inplace=False, remove_qconfig=True, is_reference=False, ...)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | mapping=None（默认映射）与 mapping={}（空映射） | 已覆盖：test_convert_empty_mapping_skips_swap |
| 枚举选项 | is_reference/remove_qconfig 布尔分支 | 已覆盖：test_convert_remove_qconfig_false_keeps_qconfig |
| 参数类型 | module 为 nn.Module，非模块输入报错 | 已覆盖：test_convert_non_module_raises |
| 传参与不传参 | inplace 显式传 True/省略默认 False | 已覆盖：test_convert_inplace_false_does_not_mutate_original |
| 等价类/边界值 | 权重 int_repr 与 scale/zero_point 反量化一致 | 已覆盖：test_convert_weight_int_repr_matches_scale_zero_point |
| 正常传参场景 | 校准后模块转换为量化模块并可前向 | 已覆盖：test_convert_returns_quantized_linear |
| 异常传参场景 | 传入非模块对象触发 AttributeError | 已覆盖：test_convert_non_module_raises |
| 混合设备类型 | 量化后端仅 CPU，无异构设备输入 | 未覆盖，原因见下 |

未覆盖项及原因：
- 混合设备类型：量化模块仅由 CPU 量化后端（qnnpack/onednn/fbgemm）提供内核，
  NPU/CUDA 上不存在量化执行路径，无法构造异构设备输入，故不做 CPU/NPU Golden 对照。
- 图模式：本用例针对 eager 转换流程，未覆盖 torch.compile / NPUGraph 图模式路径。
- 空张量：convert 作用于模块而非张量，空张量不在本接口的输入契约内，故不单列。

注意：量化后端仅 CPU 提供内核，本用例在 setUp 中按可用引擎探测选择 qnnpack/onednn/fbgemm，
     并以量化权重 int_repr 与 scale/zero_point 的反量化重建作为数值基线。
"""
import os
import unittest
import warnings

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

_ENGINE_CANDIDATES = ("qnnpack", "onednn", "fbgemm")
_SHAPE = (2, 3, 8, 8)


def _use_quantized_engine(probe=None):
    """Select a CPU quantized engine able to run the target op.

    The quantized modules only provide kernels on the CPU quantized backend, so
    there is no NPU/CUDA execution path to compare against.
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


def _quantize(x, scale=0.05, zero_point=0):
    return torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)


def _build_prepared_conv_relu():
    class _Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, 3)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            return self.relu(self.conv(x))

    net = _Net().eval()
    net.qconfig = torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine)
    torch.ao.quantization.fuse_modules(net, [["conv", "relu"]], inplace=True)
    prepared = torch.ao.quantization.prepare(net)
    prepared(torch.randn(*_SHAPE))
    return prepared


def _build_prepared_linear():
    class _Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 3)

        def forward(self, x):
            return self.lin(x)

    net = _Net().eval()
    net.qconfig = torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine)
    prepared = torch.ao.quantization.prepare(net)
    prepared(torch.randn(2, 4))
    return prepared


def _probe():
    converted = torch.ao.quantization.convert(_build_prepared_conv_relu())
    converted(_quantize(torch.randn(*_SHAPE)))


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestTorchAoQuantizationConvertUt(TestCase):
    """Functional tests for torch.ao.quantization.convert (CPU quantized backend only)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        _use_quantized_engine(_probe)

    def test_convert_returns_quantized_linear(self):
        """A calibrated Linear is swapped for the quantized Linear with qint8 weights."""
        converted = torch.ao.quantization.convert(_build_prepared_linear())
        self.assertEqual(type(converted.lin).__module__, "torch.ao.nn.quantized.modules.linear")
        self.assertEqual(converted.lin.weight().dtype, torch.qint8)

    def test_convert_intrinsic_conv_relu(self):
        """A fused Conv+ReLU is swapped for the intrinsic quantized ConvReLU2d."""
        converted = torch.ao.quantization.convert(_build_prepared_conv_relu())
        self.assertEqual(
            type(converted.conv).__module__,
            "torch.ao.nn.intrinsic.quantized.modules.conv_relu",
        )
        self.assertEqual(converted.conv.weight().dtype, torch.qint8)
        out = converted(_quantize(torch.randn(*_SHAPE)))
        self.assertEqual(out.dtype, torch.quint8)
        self.assertEqual(tuple(out.shape), (2, 4, 6, 6))

    def test_convert_output_runs_forward(self):
        """The converted module consumes a quantized input and returns a quantized output."""
        converted = torch.ao.quantization.convert(_build_prepared_linear())
        out = converted(_quantize(torch.randn(2, 4)))
        self.assertTrue(out.is_quantized)
        self.assertEqual(out.dtype, torch.quint8)
        self.assertEqual(tuple(out.shape), (2, 3))

    def test_convert_removes_qconfig_by_default(self):
        """remove_qconfig defaults to True and strips qconfig from the converted module."""
        prepared = _build_prepared_linear()
        self.assertTrue(hasattr(prepared, "qconfig"))
        converted = torch.ao.quantization.convert(prepared)
        self.assertFalse(hasattr(converted, "qconfig"))

    def test_convert_remove_qconfig_false_keeps_qconfig(self):
        """With remove_qconfig=False the qconfig attribute survives the conversion."""
        converted = torch.ao.quantization.convert(
            _build_prepared_linear(), remove_qconfig=False)
        self.assertTrue(hasattr(converted, "qconfig"))

    def test_convert_inplace_true_returns_same_object(self):
        """inplace=True mutates and returns the very same module object."""
        prepared = _build_prepared_linear()
        converted = torch.ao.quantization.convert(prepared, inplace=True)
        self.assertIs(converted, prepared)

    def test_convert_inplace_false_does_not_mutate_original(self):
        """inplace=False deep-copies: the caller's module stays float and keeps its qconfig."""
        prepared = _build_prepared_linear()
        converted = torch.ao.quantization.convert(prepared)
        self.assertIsNot(converted, prepared)
        self.assertTrue(hasattr(prepared, "qconfig"))
        self.assertEqual(type(prepared.lin).__module__, "torch.nn.modules.linear")
        self.assertEqual(type(converted.lin).__module__, "torch.ao.nn.quantized.modules.linear")

    def test_convert_empty_mapping_skips_swap(self):
        """An explicit empty mapping disables every swap, leaving the float module intact."""
        converted = torch.ao.quantization.convert(_build_prepared_linear(), mapping={})
        self.assertEqual(type(converted.lin).__module__, "torch.nn.modules.linear")

    def test_convert_is_reference_produces_reference_module(self):
        """is_reference=True maps to the float reference module instead of the quantized one."""
        converted = torch.ao.quantization.convert(_build_prepared_linear(), is_reference=True)
        self.assertEqual(
            type(converted.lin).__module__,
            "torch.ao.nn.quantized.reference.modules.linear",
        )
        self.assertEqual(converted.lin.weight.dtype, torch.float32)

    def test_convert_weight_int_repr_matches_scale_zero_point(self):
        """The packed weight int_repr dequantizes back to the original float weight."""
        prepared = _build_prepared_linear()
        float_weight = prepared.lin.weight.detach().clone()
        converted = torch.ao.quantization.convert(prepared)
        weight = converted.lin.weight()
        int_weight = weight.int_repr()
        self.assertEqual(int_weight.dtype, torch.int8)
        self.assertEqual(tuple(int_weight.shape), tuple(float_weight.shape))
        scale = float(weight.q_scale())
        zero_point = int(weight.q_zero_point())
        dequant = (int_weight.float() - zero_point) * scale
        torch.testing.assert_close(dequant, float_weight, atol=scale, rtol=0)

    def test_convert_non_module_raises(self):
        """Passing a non-module object raises AttributeError from the traversal."""
        with self.assertRaises(AttributeError):
            torch.ao.quantization.convert(123)

    def test_convert_emits_deprecation_warning(self):
        """convert is deprecated and must emit a DeprecationWarning on every call."""
        prepared = _build_prepared_linear()
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            torch.ao.quantization.convert(prepared)
        categories = [record.category for record in records]
        self.assertTrue(any(issubclass(cat, DeprecationWarning) for cat in categories))


if __name__ == "__main__":
    run_tests()
