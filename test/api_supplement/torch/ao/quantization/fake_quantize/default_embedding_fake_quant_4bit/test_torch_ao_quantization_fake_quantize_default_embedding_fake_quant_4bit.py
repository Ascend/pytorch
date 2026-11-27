# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.fake_quantize.default_embedding_fake_quant_4bit 接口功能正确性
API 名称：torch.ao.quantization.fake_quantize.default_embedding_fake_quant_4bit
API 签名：default_embedding_fake_quant_4bit() -> FakeQuantize（工厂，partial 包装）

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | forward 输入非空浮点张量 | 已覆盖：test_forward_preserves_shape |
| 枚举选项 | quint4x2 的 4bit 量化范围 [0,15] | 已覆盖：test_quant_range_is_4bit |
| 参数类型 | 工厂实例化无参、forward 接受浮点 Tensor | 已覆盖：test_returns_fake_quantize_instance |
| 传参与不传参 | 工厂实例化不传参，使用默认观察者参数 | 已覆盖：test_returns_fake_quantize_instance |
| 等价类/边界值 | 每通道 scale 数量等于 ch_axis 维大小 | 已覆盖：test_forward_per_channel_scale_shape |
| 正常传参场景 | forward 输出落在 4bit 每通道量化栅格 | 已覆盖：test_forward_output_on_quant_grid |
| 异常传参场景 | 工厂不接受参数，无稳定异常路径 | 未覆盖，原因见下 |
| 混合设备类型 | forward 为单 Tensor 输入 | 未覆盖，原因见下 |

未覆盖项及原因：
- 异常传参场景：default_embedding_fake_quant_4bit 为无参工厂，不存在可稳定断言的非法入参路径。
- 混合设备类型：该伪造量化模块的 forward 仅接受单个 Tensor，无多 Tensor 异构设备输入场景。
- 空张量：观察者需依据输入统计 min/max，空张量使统计未定义，故不单列边界用例。

注意：本模块的量化后端仅 CPU 提供内核，且伪造量化为纯张量运算，
     无 NPU/CUDA 对照路径，故不做 CPU/NPU Golden 对照；
     以量化公式 torch.fake_quantize_per_channel_affine 的重建结果作为数值基线。
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

from torch.ao.quantization.fake_quantize import FakeQuantize, default_embedding_fake_quant_4bit


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestDefaultEmbeddingFakeQuant4bitUt(TestCase):
    """Functional tests for default_embedding_fake_quant_4bit (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_returns_fake_quantize_instance(self):
        """Calling the factory builds a FakeQuantize; calls yield distinct instances."""
        module = default_embedding_fake_quant_4bit()
        other = default_embedding_fake_quant_4bit()
        self.assertIsInstance(module, FakeQuantize)
        self.assertIsNot(module, other)

    def test_dtype_is_quint4x2(self):
        """The 4bit default emulates the torch.quint4x2 dtype."""
        module = default_embedding_fake_quant_4bit()
        self.assertEqual(module.dtype, torch.quint4x2)

    def test_quant_range_is_4bit(self):
        """The quant range collapses to the 4bit [0, 15] window."""
        module = default_embedding_fake_quant_4bit()
        self.assertEqual(module.quant_min, 0)
        self.assertEqual(module.quant_max, 15)
        self.assertEqual(module.activation_post_process.quant_max, 15)

    def test_qscheme_is_per_channel_float_qparams(self):
        """Embeddings use per-channel affine float qparams with ch_axis 0."""
        module = default_embedding_fake_quant_4bit()
        self.assertEqual(module.qscheme, torch.per_channel_affine_float_qparams)
        self.assertTrue(module.is_per_channel)
        self.assertEqual(module.ch_axis, 0)
        self.assertEqual(module.zero_point.dtype, torch.float32)

    def test_forward_per_channel_scale_shape(self):
        """After a forward the per-channel scale has one entry per embedding row."""
        module = default_embedding_fake_quant_4bit()
        x = torch.randn(5, 6)
        module(x)
        self.assertEqual(tuple(module.scale.shape), (5,))

    def test_forward_output_on_quant_grid(self):
        """Output equals the 4bit per-channel affine fake-quant reconstruction."""
        module = default_embedding_fake_quant_4bit()
        x = torch.randn(5, 6)
        out = module(x)
        expected = torch.fake_quantize_per_channel_affine(
            x, module.scale, module.zero_point, 0, 0, 15)
        torch.testing.assert_close(out, expected)

    def test_forward_preserves_shape(self):
        """Fake quantization keeps the tensor shape."""
        module = default_embedding_fake_quant_4bit()
        x = torch.randn(5, 6)
        self.assertEqual(tuple(module(x).shape), tuple(x.shape))


if __name__ == "__main__":
    run_tests()
