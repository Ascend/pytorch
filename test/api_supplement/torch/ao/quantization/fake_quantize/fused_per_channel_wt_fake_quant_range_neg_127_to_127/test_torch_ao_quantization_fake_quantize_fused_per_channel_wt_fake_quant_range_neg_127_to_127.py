# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.fake_quantize.fused_per_channel_wt_fake_quant_range_neg_127_to_127 功能正确性
API 名称：torch.ao.quantization.fake_quantize.fused_per_channel_wt_fake_quant_range_neg_127_to_127
API 签名：fused_per_channel_wt_fake_quant_range_neg_127_to_127() -> FusedMovingAvgObsFakeQuantize（工厂）

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | forward 输入非空浮点张量 | 已覆盖：test_forward_preserves_shape_and_scale_shape |
| 枚举选项 | per_channel_symmetric + qint8 + [-127,127] | 已覆盖：test_quant_range_and_dtype |
| 参数类型 | 工厂实例化无参、forward 接受浮点 Tensor | 已覆盖：test_returns_fused_module |
| 传参与不传参 | 工厂实例化不传参，使用默认观察者参数 | 已覆盖：test_returns_fused_module |
| 等价类/边界值 | quant_min 排除 -128、每通道 scale 数量 | 已覆盖：test_quant_min_excludes_neg128 |
| 正常传参场景 | forward 输出落在每通道对称量化栅格 | 已覆盖：test_forward_output_on_quant_grid |
| 异常传参场景 | 工厂不接受参数，无稳定异常路径 | 未覆盖，原因见下 |
| 混合设备类型 | forward 为单 Tensor 输入 | 未覆盖，原因见下 |

未覆盖项及原因：
- 异常传参场景：该融合伪造量化工厂为无参工厂，不存在可稳定断言的非法入参路径。
- 混合设备类型：该伪造量化模块的 forward 仅接受单个 Tensor，无多 Tensor 异构设备输入场景。
- 空张量：融合观察者需依据输入统计 min/max，空张量使统计未定义，故不单列边界用例。

注意：融合伪造量化内核仅在 CPU 量化后端（qnnpack/onednn/fbgemm）可用，
     NPU/CUDA 上无对应执行路径，故不做 CPU/NPU Golden 对照；
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

from torch.ao.quantization.fake_quantize import (
    FusedMovingAvgObsFakeQuantize,
    fused_per_channel_wt_fake_quant_range_neg_127_to_127,
)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestFusedPerChannelWtFakeQuantRangeNeg127To127Ut(TestCase):
    """Functional tests for fused_per_channel_wt_fake_quant_range_neg_127_to_127 (CPU kernels only)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_returns_fused_module(self):
        """The factory builds a FusedMovingAvgObsFakeQuantize; calls yield distinct instances."""
        module = fused_per_channel_wt_fake_quant_range_neg_127_to_127()
        other = fused_per_channel_wt_fake_quant_range_neg_127_to_127()
        self.assertIsInstance(module, FusedMovingAvgObsFakeQuantize)
        self.assertIsNot(module, other)

    def test_per_channel_symmetric_flags(self):
        """Per-channel symmetric quantization is signalled via the public flags."""
        module = fused_per_channel_wt_fake_quant_range_neg_127_to_127()
        self.assertTrue(module.is_per_channel)
        self.assertTrue(module.is_symmetric_quant)
        self.assertEqual(module.ch_axis, 0)

    def test_quant_range_and_dtype(self):
        """Quant range, dtype and qscheme match the documented defaults."""
        module = fused_per_channel_wt_fake_quant_range_neg_127_to_127()
        self.assertEqual(module.quant_min, -127)
        self.assertEqual(module.quant_max, 127)
        self.assertEqual(module.dtype, torch.qint8)
        self.assertEqual(module.qscheme, torch.per_channel_symmetric)

    def test_quant_min_excludes_neg128(self):
        """The range deliberately excludes -128, unlike the plain fused weight defaults."""
        module = fused_per_channel_wt_fake_quant_range_neg_127_to_127()
        self.assertEqual(module.quant_min, -127)
        self.assertNotEqual(module.quant_min, -128)

    def test_forward_preserves_shape_and_scale_shape(self):
        """Fake quantization keeps the input shape and yields one scale per channel."""
        module = fused_per_channel_wt_fake_quant_range_neg_127_to_127()
        x = torch.randn(4, 5)
        out = module(x)
        self.assertEqual(tuple(out.shape), tuple(x.shape))
        self.assertEqual(tuple(module.scale.shape), (4,))

    def test_forward_output_on_quant_grid(self):
        """Output equals the per-channel symmetric fake-quant reconstruction of the input."""
        module = fused_per_channel_wt_fake_quant_range_neg_127_to_127()
        x = torch.randn(4, 5)
        out = module(x)
        expected = torch.fake_quantize_per_channel_affine(
            x, module.scale, torch.zeros_like(module.zero_point), 0, -127, 127)
        torch.testing.assert_close(out, expected)

    def test_extra_repr_contains_reduce_range(self):
        """The fused module repr exposes the observer's reduce_range setting."""
        module = fused_per_channel_wt_fake_quant_range_neg_127_to_127()
        self.assertIn("reduce_range=", module.extra_repr())
        self.assertIn("quant_min=-127", module.extra_repr())


if __name__ == "__main__":
    run_tests()
