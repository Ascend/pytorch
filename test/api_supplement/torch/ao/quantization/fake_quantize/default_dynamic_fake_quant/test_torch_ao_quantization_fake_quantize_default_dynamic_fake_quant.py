# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.fake_quantize.default_dynamic_fake_quant 接口功能正确性
API 名称：torch.ao.quantization.fake_quantize.default_dynamic_fake_quant
API 签名：default_dynamic_fake_quant() -> FakeQuantize（工厂，partial 包装）

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | forward 输入非空浮点张量 | 已覆盖：test_forward_preserves_shape_and_dtype |
| 枚举选项 | dtype=quint8 的量化范围与 qscheme | 已覆盖：test_quant_range_and_dtype |
| 参数类型 | 工厂实例化无参、forward 接受浮点 Tensor | 已覆盖：test_returns_fake_quantize_instance |
| 传参与不传参 | 工厂实例化不传参，使用默认观察者参数 | 已覆盖：test_returns_fake_quantize_instance |
| 等价类/边界值 | 关断 fake_quant 后 forward 恒等 | 已覆盖：test_forward_identity_when_fake_quant_disabled |
| 正常传参场景 | forward 输出落在量化栅格且形状不变 | 已覆盖：test_forward_output_on_quant_grid |
| 异常传参场景 | 工厂不接受参数，无稳定异常路径 | 未覆盖，原因见下 |
| 混合设备类型 | forward 为单 Tensor 输入 | 未覆盖，原因见下 |

未覆盖项及原因：
- 异常传参场景：default_dynamic_fake_quant 为无参工厂，实例化与配置校验均在内部固定，
  不存在可稳定断言的非法入参路径。
- 混合设备类型：该伪造量化模块的 forward 仅接受单个 Tensor，无多 Tensor 异构设备输入场景。
- 空张量：观察者需依据输入统计 min/max，空张量使统计未定义，故不单列边界用例。

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

from torch.ao.quantization.fake_quantize import FakeQuantize, default_dynamic_fake_quant


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestDefaultDynamicFakeQuantUt(TestCase):
    """Functional tests for default_dynamic_fake_quant (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_returns_fake_quantize_instance(self):
        """Calling the factory builds a FakeQuantize; calls yield distinct instances."""
        module = default_dynamic_fake_quant()
        other = default_dynamic_fake_quant()
        self.assertIsInstance(module, FakeQuantize)
        self.assertIsNot(module, other)

    def test_is_dynamic_flag(self):
        """The default marks both the module and its observer as dynamic."""
        module = default_dynamic_fake_quant()
        self.assertTrue(module.is_dynamic)
        self.assertTrue(module.activation_post_process.is_dynamic)

    def test_quant_range_and_dtype(self):
        """Quant range, dtype and qscheme match the documented defaults."""
        module = default_dynamic_fake_quant()
        self.assertEqual(module.quant_min, 0)
        self.assertEqual(module.quant_max, 255)
        self.assertEqual(module.activation_post_process.quant_min, 0)
        self.assertEqual(module.activation_post_process.quant_max, 255)
        self.assertEqual(module.dtype, torch.quint8)
        self.assertEqual(module.qscheme, torch.per_tensor_affine)

    def test_forward_preserves_shape_and_dtype(self):
        """Fake quantization keeps the tensor shape and float dtype."""
        module = default_dynamic_fake_quant()
        x = torch.randn(4, 6)
        out = module(x)
        self.assertEqual(tuple(out.shape), tuple(x.shape))
        self.assertEqual(out.dtype, torch.float32)

    def test_forward_output_on_quant_grid(self):
        """Output equals the per-tensor affine fake-quant reconstruction of the input."""
        module = default_dynamic_fake_quant()
        x = torch.randn(4, 6) * 3.0
        out = module(x)
        expected = torch.fake_quantize_per_tensor_affine(
            x, module.scale, module.zero_point.to(torch.int32), 0, 255)
        torch.testing.assert_close(out, expected)

    def test_forward_identity_when_fake_quant_disabled(self):
        """After disable_fake_quant the module returns the input unchanged."""
        module = default_dynamic_fake_quant()
        module.disable_fake_quant()
        x = torch.randn(4, 6)
        out = module(x)
        torch.testing.assert_close(out, x)


if __name__ == "__main__":
    run_tests()
