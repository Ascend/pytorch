# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.fake_quantize.FixedQParamsFakeQuantize.extra_repr 接口功能正确性
API 名称：torch.ao.quantization.fake_quantize.FixedQParamsFakeQuantize.extra_repr
API 签名：FixedQParamsFakeQuantize.extra_repr(self) -> str

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | 该方法无输入参数 | 不适用 |
| 枚举选项 | 固定观察者的 scale/zero_point 组合 | 已覆盖：test_extra_repr_reflects_fixed_scale_and_zero_point |
| 参数类型 | 该方法不接受任何参数 | 不适用 |
| 传参与不传参 | 该方法无可选参数 | 不适用 |
| 等价类/边界值 | fake_quant/observer 开关的 0/1 状态 | 已覆盖：test_extra_repr_reflects_toggle_state |
| 正常传参场景 | 返回包含全部属性的 repr 字符串 | 已覆盖：test_extra_repr_returns_attribute_string |
| 异常传参场景 | 该方法不接受参数，无稳定异常路径 | 未覆盖，原因见下 |
| 混合设备类型 | 纯字符串格式化，不涉及设备迁移 | 未覆盖，原因见下 |

未覆盖项及原因：
- 异常传参场景：extra_repr 为无参方法，不存在可稳定断言的非法入参路径。
- 混合设备类型：该方法仅做字符串格式化，不接收 Tensor、不涉及设备迁移，无异构设备输入可构造。
- 空/非空、参数类型、传参与不传参：该方法不接收任何输入，故标注不适用。

注意：本接口无 Tensor 数值输出，不适用 CPU/NPU Golden 对照；
     按其返回字符串的内容与模块 repr 的属性一致性进行断言。
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

from torch.ao.quantization.fake_quantize import FixedQParamsFakeQuantize
from torch.ao.quantization.observer import (
    default_fixed_qparams_range_0to1_observer,
    default_fixed_qparams_range_neg1to1_observer,
)


def _build_fixed(observer):
    """Build a FixedQParamsFakeQuantize bound to the given fixed observer."""
    return FixedQParamsFakeQuantize(observer=observer)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestFixedqparamsFakequantizeExtrareprUt(TestCase):
    """Functional tests for FixedQParamsFakeQuantize.extra_repr (pure Python, NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_extra_repr_returns_attribute_string(self):
        """extra_repr returns a string listing every documented attribute key."""
        module = _build_fixed(default_fixed_qparams_range_neg1to1_observer)
        result = module.extra_repr()
        self.assertIsInstance(result, str)
        for key in (
            "fake_quant_enabled=",
            "observer_enabled=",
            "scale=",
            "zero_point=",
            "dtype=",
            "quant_min=",
            "quant_max=",
            "qscheme=",
        ):
            self.assertIn(key, result)

    def test_extra_repr_reflects_fixed_scale_and_zero_point(self):
        """The fixed scale/zero_point of the bound observer appear in the repr."""
        neg = _build_fixed(default_fixed_qparams_range_neg1to1_observer).extra_repr()
        zero = _build_fixed(default_fixed_qparams_range_0to1_observer).extra_repr()
        self.assertIn("scale=tensor([0.0078])", neg)
        self.assertIn("zero_point=tensor([128], dtype=torch.int32)", neg)
        self.assertIn("scale=tensor([0.0039])", zero)
        self.assertIn("zero_point=tensor([0], dtype=torch.int32)", zero)
        self.assertNotEqual(neg, zero)

    def test_extra_repr_reflects_toggle_state(self):
        """Toggling fake_quant/observer enable buffers changes the rendered values."""
        module = _build_fixed(default_fixed_qparams_range_neg1to1_observer)
        self.assertIn("fake_quant_enabled=tensor([1], dtype=torch.uint8)", module.extra_repr())
        module.disable_fake_quant()
        self.assertIn("fake_quant_enabled=tensor([0], dtype=torch.uint8)", module.extra_repr())
        module.enable_fake_quant()
        self.assertIn("fake_quant_enabled=tensor([1], dtype=torch.uint8)", module.extra_repr())
        module.disable_observer()
        self.assertIn("observer_enabled=tensor([0], dtype=torch.uint8)", module.extra_repr())

    def test_extra_repr_reflects_quant_range_and_dtype(self):
        """The quant range and dtype of the fixed observer are rendered verbatim."""
        module = _build_fixed(default_fixed_qparams_range_neg1to1_observer)
        result = module.extra_repr()
        self.assertIn("dtype=torch.quint8", result)
        self.assertIn("quant_min=0", result)
        self.assertIn("quant_max=255", result)
        self.assertIn("qscheme=torch.per_tensor_affine", result)

    def test_extra_repr_embedded_in_module_repr(self):
        """nn.Module.__repr__ embeds the string returned by extra_repr."""
        module = _build_fixed(default_fixed_qparams_range_neg1to1_observer)
        self.assertIn(module.extra_repr(), repr(module))


if __name__ == "__main__":
    run_tests()
