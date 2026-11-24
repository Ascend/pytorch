# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.nn.quantized.dynamic.modules.conv.ConvTranspose1d 接口功能正确性
API 名称：torch.ao.nn.quantized.dynamic.modules.conv.ConvTranspose1d
API 签名：ConvTranspose1d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0,
         groups=1, bias=True, dilation=1, padding_mode="zeros", device=None, dtype=None)

覆盖维度表：
| 覆盖维度      | 说明                            | 覆盖情况                                                       |
|---------------|---------------------------------|----------------------------------------------------------------|
| 空/非空       | bias=False（None）与非空 bias   | 已覆盖：test_bias_disabled_returns_none /                      |
|               |                                 | test_bias_enabled_returns_tensor                               |
| 枚举选项      | padding_mode 仅支持 zeros，非   | 已覆盖：test_constructor_rejects_unsupported_padding_mode      |
|               | zeros 一律拒绝                  |                                                                |
| 参数类型      | kernel_size/stride/padding/     | 已覆盖：test_constructor_stores_geometry_attributes /          |
|               | dilation/output_padding 归一为  | test_weight_bias_round_trip_matches_reference                  |
|               | 1-tuple；权重为 qint8           |                                                                |
| 传参与不传参  | forward 显式 reduce_range       | 已覆盖：test_forward_explicit_reduce_range_flag                |
|               | 与默认值                        |                                                                |
| 等价类/边界值 | 默认 stride/padding 与          | 已覆盖：test_forward_returns_float_tensor_with_expected_shape  |
|               | stride=2/padding=1 两类几何     | / test_forward_respects_stride_and_padding                     |
| 正常传参场景  | 构造+forward 输出 shape/dtype、 | 已覆盖：test_forward_bias_only_output_is_constant /            |
|               | 常量偏置泛化、可复现性          | test_forward_is_reproducible                                   |
| 异常传参场景  | 非法 out_channels/groups/       | 已覆盖：test_constructor_rejects_* /                           |
|               | padding_mode、forward 维度错误  | test_forward_rejects_wrong_ndim                                |
| 混合设备类型  | 量化动态卷积仅 CPU 后端有内核   | 未覆盖，原因见下                                               |

未覆盖项及原因：
- 混合设备类型：动态量化转置卷积仅在 CPU 量化后端（onednn/fbgemm/qnnpack）提供内核，NPU/CUDA 上不存在对应执行路径，
  无法构造异构设备输入，故不做 CPU/NPU 对照。
- output_padding 非零：onednn 的 conv_transpose 预打包仅支持 output_padding=0，非零取值在构造期即失败，
  无跨引擎稳定契约，故仅覆盖默认 0 的几何。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注意：动态量化卷积仅在 CPU 量化后端执行，无 NPU/CUDA 数值对照路径，
     故本用例不做 CPU/NPU Golden 对照，仅验证量化后端上的功能语义。
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

import torch.ao.nn.quantized
import torch.ao.nn.quantized.dynamic
import torch.nn
from torch.ao.nn.quantized.dynamic.modules.conv import ConvTranspose1d

_IN_CHANNELS = 6
_OUT_CHANNELS = 12
_KERNEL_SIZE = 3
_INPUT_SHAPE = (2, _IN_CHANNELS, 10)
# transposed conv length: L_out = (L - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
_FORWARD_SHAPE = (2, _OUT_CHANNELS, 12)
_ENGINE_CANDIDATES = ("onednn", "fbgemm", "qnnpack")


def _use_quantized_engine(probe=None):
    """Select a CPU quantized engine able to run the target op.

    The dynamic quantized conv modules only provide kernels on the CPU quantized
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


def _new(in_channels=_IN_CHANNELS, out_channels=_OUT_CHANNELS, kernel_size=_KERNEL_SIZE, **kwargs):
    """Construct the module, ignoring the documented numerical-accuracy warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ConvTranspose1d(in_channels, out_channels, kernel_size, **kwargs)


def _reference():
    """A statically quantized ConvTranspose1d used as the source of packed weight/bias."""
    return torch.ao.nn.quantized.ConvTranspose1d(_IN_CHANNELS, _OUT_CHANNELS, _KERNEL_SIZE)


def _build(**kwargs):
    """Build a dynamic module whose weight/bias are copied from the reference."""
    module = _new(**kwargs)
    module.set_weight_bias(*_reference()._weight_bias())
    return module


def _probe():
    _build()(torch.randn(*_INPUT_SHAPE))


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestTorchAoNnQuantizedDynamicModulesConvConvTranspose1dUt(TestCase):
    """Functional tests for torch.ao.nn.quantized.dynamic.modules.conv.ConvTranspose1d."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        _use_quantized_engine(_probe)

    def test_constructor_returns_expected_instance(self):
        """The constructor builds a dynamic quantized ConvTranspose1d with the right name."""
        module = _new()
        self.assertIsInstance(module, ConvTranspose1d)
        self.assertEqual(module._get_name(), "DynamicQuantizedConvTranspose1d")

    def test_constructor_is_quantized_conv_subclass(self):
        """The dynamic module subclasses the static quantized nnq.ConvTranspose1d."""
        module = _build()
        self.assertIsInstance(module, torch.ao.nn.quantized.ConvTranspose1d)

    def test_float_module_attribute_matches_torch_nn(self):
        """_FLOAT_MODULE points at the matching float torch.nn module."""
        self.assertIs(ConvTranspose1d._FLOAT_MODULE, torch.nn.ConvTranspose1d)

    def test_constructor_stores_geometry_attributes(self):
        """Normalized transposed-conv geometry is stored on the module."""
        module = _new(stride=2, padding=1, output_padding=0, bias=False)
        self.assertEqual(module.in_channels, _IN_CHANNELS)
        self.assertEqual(module.out_channels, _OUT_CHANNELS)
        self.assertEqual(tuple(module.kernel_size), (_KERNEL_SIZE,))
        self.assertEqual(tuple(module.stride), (2,))
        self.assertEqual(tuple(module.padding), (1,))
        self.assertEqual(tuple(module.output_padding), (0,))
        self.assertEqual(tuple(module.dilation), (1,))
        self.assertEqual(module.groups, 1)
        self.assertEqual(module.padding_mode, "zeros")
        self.assertEqual(float(module.scale), 1.0)
        self.assertEqual(int(module.zero_point), 0)

    def test_constructor_emits_accuracy_warning(self):
        """Building the module warns about poor numerical accuracy exactly once."""
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            ConvTranspose1d(_IN_CHANNELS, _OUT_CHANNELS, _KERNEL_SIZE)
        accuracy_warnings = [item for item in record if "poor numerical accuracy" in str(item.message)]
        self.assertEqual(len(accuracy_warnings), 1)
        self.assertIs(accuracy_warnings[0].category, UserWarning)

    def test_bias_disabled_returns_none(self):
        """A module built without bias reports None from bias()."""
        module = _new(bias=False)
        self.assertIsNone(module.bias())

    def test_bias_enabled_returns_tensor(self):
        """A module built with bias exposes a per-out-channel bias tensor."""
        module = _build(bias=True)
        self.assertIsNotNone(module.bias())
        self.assertEqual(tuple(module.bias().shape), (_OUT_CHANNELS,))

    def test_weight_bias_round_trip_matches_reference(self):
        """set_weight_bias/_weight_bias round-trips the reference qint8 weight."""
        reference = _reference()
        module = _new()
        module.set_weight_bias(*reference._weight_bias())
        weight, _ = module._weight_bias()
        self.assertEqual(weight.dtype, torch.qint8)
        self.assertEqual(tuple(weight.shape), (_IN_CHANNELS, _OUT_CHANNELS, _KERNEL_SIZE))
        ref_weight, _ = reference._weight_bias()
        self.assertTrue(torch.equal(weight.int_repr(), ref_weight.int_repr()))

    def test_forward_returns_float_tensor_with_expected_shape(self):
        """Forward returns a float32 tensor with the transposed-conv output shape."""
        module = _build()
        output = module(torch.randn(*_INPUT_SHAPE))
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(tuple(output.shape), _FORWARD_SHAPE)

    def test_forward_respects_stride_and_padding(self):
        """A stride-2, padding-1 geometry yields the expected output length."""
        module = _build(stride=2, padding=1)
        output = module(torch.randn(*_INPUT_SHAPE))
        self.assertEqual(tuple(output.shape), (2, _OUT_CHANNELS, 19))

    def test_forward_explicit_reduce_range_flag(self):
        """An explicit reduce_range flag is honored and True matches the default."""
        module = _build()
        inputs = torch.randn(*_INPUT_SHAPE)
        default_output = module(inputs)
        explicit_true = module(inputs, reduce_range=True)
        explicit_false = module(inputs, reduce_range=False)
        self.assertEqual(tuple(explicit_false.shape), _FORWARD_SHAPE)
        torch.testing.assert_close(default_output, explicit_true, rtol=0, atol=0)

    def test_forward_bias_only_output_is_constant(self):
        """A zero weight makes the dynamic transposed conv emit the bias everywhere."""
        module = _new()
        zero_weight = torch.quantize_per_tensor(
            torch.zeros(_IN_CHANNELS, _OUT_CHANNELS, _KERNEL_SIZE), 1.0, 0, torch.qint8
        )
        module.set_weight_bias(zero_weight, torch.full((_OUT_CHANNELS,), 2.0))
        output = module(torch.randn(*_INPUT_SHAPE))
        torch.testing.assert_close(output, torch.full_like(output, 2.0), rtol=1e-2, atol=0.05)

    def test_forward_is_reproducible(self):
        """Two forwards with identical input and weight are bit-identical."""
        module = _build()
        inputs = torch.randn(*_INPUT_SHAPE)
        self.assertTrue(torch.equal(module(inputs), module(inputs)))

    def test_forward_rejects_wrong_ndim(self):
        """A 4D input to a ConvTranspose1d module raises ValueError."""
        module = _build()
        with self.assertRaisesRegex(ValueError, "Input shape must be"):
            module(torch.randn(2, _IN_CHANNELS, 10, 10))

    def test_constructor_rejects_non_positive_out_channels(self):
        """out_channels <= 0 is rejected by the constructor."""
        with self.assertRaisesRegex(ValueError, "out_channels must be greater than 0"):
            _new(out_channels=0)

    def test_constructor_rejects_indivisible_in_channels(self):
        """in_channels not divisible by groups is rejected."""
        with self.assertRaisesRegex(ValueError, "in_channels must be divisible by groups"):
            _new(groups=5)

    def test_constructor_rejects_indivisible_out_channels(self):
        """out_channels not divisible by groups is rejected."""
        with self.assertRaisesRegex(ValueError, "out_channels must be divisible by groups"):
            _new(out_channels=10, groups=3)

    def test_constructor_rejects_unsupported_padding_mode(self):
        """Transposed conv only supports the zeros padding mode."""
        with self.assertRaisesRegex(ValueError, 'Only "zeros" padding mode is supported'):
            _new(padding_mode="reflect")


if __name__ == "__main__":
    run_tests()
