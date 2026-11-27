# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.fake_quantize.default_fixed_qparams_range_neg1to1_fake_quant 接口功能正确性
API 名称：torch.ao.quantization.fake_quantize.default_fixed_qparams_range_neg1to1_fake_quant
API 签名：default_fixed_qparams_range_neg1to1_fake_quant() -> FixedQParamsFakeQuantize（工厂）

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | forward 输入非空浮点张量 | 已覆盖：test_forward_output_on_quant_grid |
| 枚举选项 | 固定 scale=2/256、zero_point=128 组合 | 已覆盖：test_fixed_scale_and_zero_point |
| 参数类型 | 工厂实例化无参、forward 接受浮点 Tensor | 已覆盖：test_returns_fixed_qparams_fake_quantize |
| 传参与不传参 | 工厂实例化不传参，使用固定观察者 | 已覆盖：test_observer_type_is_fixed_qparams |
| 等价类/边界值 | 输入超出 [-1,1] 时饱和到固定范围端点 | 已覆盖：test_forward_saturates_out_of_range |
| 正常传参场景 | forward 输出落在固定量化栅格 | 已覆盖：test_forward_output_on_quant_grid |
| 异常传参场景 | 工厂不接受参数，无稳定异常路径 | 未覆盖，原因见下 |
| 混合设备类型 | forward 为单 Tensor 输入 | 未覆盖，原因见下 |

未覆盖项及原因：
- 异常传参场景：default_fixed_qparams_range_neg1to1_fake_quant 为无参工厂，不存在可稳定断言的非法入参路径。
- 混合设备类型：该伪造量化模块的 forward 仅接受单个 Tensor，无多 Tensor 异构设备输入场景。
- 空张量：固定量化参数的 forward 虽不依赖统计，但空张量输出无观测价值，故不单列边界用例。

注意：本模块的量化后端仅 CPU 提供内核，且伪造量化为纯张量运算，
     无 NPU/CUDA 对照路径，故不做 CPU/NPU Golden 对照；
     以量化公式 torch.fake_quantize_per_tensor_affine 的重建结果作为数值基线。
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
    FixedQParamsFakeQuantize,
    default_fixed_qparams_range_neg1to1_fake_quant,
)
from torch.ao.quantization.observer import FixedQParamsObserver


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestDefaultFixedQparamsRangeNeg1to1FakeQuantUt(TestCase):
    """Functional tests for default_fixed_qparams_range_neg1to1_fake_quant (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_returns_fixed_qparams_fake_quantize(self):
        """The factory builds a FixedQParamsFakeQuantize; calls yield distinct instances."""
        module = default_fixed_qparams_range_neg1to1_fake_quant()
        other = default_fixed_qparams_range_neg1to1_fake_quant()
        self.assertIsInstance(module, FixedQParamsFakeQuantize)
        self.assertIsNot(module, other)

    def test_observer_type_is_fixed_qparams(self):
        """FixedQParamsFakeQuantize requires its activation observer to be FixedQParamsObserver."""
        module = default_fixed_qparams_range_neg1to1_fake_quant()
        self.assertIsInstance(module.activation_post_process, FixedQParamsObserver)

    def test_fixed_scale_and_zero_point(self):
        """The fixed scale is 2/256 and the fixed zero point is 128."""
        module = default_fixed_qparams_range_neg1to1_fake_quant()
        self.assertAlmostEqual(float(module.scale), 2.0 / 256.0, places=6)
        self.assertEqual(int(module.zero_point), 128)

    def test_quant_range_and_dtype(self):
        """Quant range, dtype and qscheme match the documented defaults."""
        module = default_fixed_qparams_range_neg1to1_fake_quant()
        self.assertEqual(module.quant_min, 0)
        self.assertEqual(module.quant_max, 255)
        self.assertEqual(module.dtype, torch.quint8)
        self.assertEqual(module.qscheme, torch.per_tensor_affine)

    def test_forward_output_on_quant_grid(self):
        """Output equals the fixed-params per-tensor affine fake-quant reconstruction."""
        module = default_fixed_qparams_range_neg1to1_fake_quant()
        x = torch.linspace(-1.0, 1.0, 11)
        out = module(x)
        expected = torch.fake_quantize_per_tensor_affine(
            x, module.scale, module.zero_point.to(torch.int32), 0, 255)
        torch.testing.assert_close(out, expected)

    def test_forward_saturates_out_of_range(self):
        """Inputs outside [-1, 1] saturate to the fixed range endpoints."""
        module = default_fixed_qparams_range_neg1to1_fake_quant()
        x = torch.tensor([-10.0, 10.0])
        out = module(x)
        self.assertAlmostEqual(float(out[0]), -1.0, places=6)
        self.assertAlmostEqual(float(out[1]), 127.0 * (2.0 / 256.0), places=6)

    def test_fixed_qparams_are_input_independent(self):
        """The fixed scale/zero_point never change across forward calls with different inputs."""
        module = default_fixed_qparams_range_neg1to1_fake_quant()
        scale0 = float(module.scale)
        zero_point0 = int(module.zero_point)
        module(torch.randn(8) * 100.0)
        module(torch.zeros(4))
        self.assertAlmostEqual(float(module.scale), scale0, places=6)
        self.assertEqual(int(module.zero_point), zero_point0)


if __name__ == "__main__":
    run_tests()
