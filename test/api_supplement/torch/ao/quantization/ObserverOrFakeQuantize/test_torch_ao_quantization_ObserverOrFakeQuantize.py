# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.ObserverOrFakeQuantize 类型别名接口正确性
API 名称：torch.ao.quantization.ObserverOrFakeQuantize
API 签名：ObserverOrFakeQuantize = TypeAliasType("ObserverOrFakeQuantize", ObserverBase | FakeQuantizeBase)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | 纯类型别名，本身不可实例化，无空/非空输入 | 不适用 |
| 枚举选项 | 该类型别名无枚举参数 | 不适用 |
| 参数类型 | 别名联合成员为 ObserverBase 与 FakeQuantizeBase | 已覆盖：test_value_union_members |
| 传参与不传参 | 类型别名无构造参数 | 不适用 |
| 等价类/边界值 | 别名可作为 isinstance 判据（__value__ 联合） | 已覆盖：test_isinstance_matches_observers_and_fake_quant |
| 正常传参场景 | observer 与 fake-quant 实例均属该别名 | 已覆盖：test_isinstance_matches_observers_and_fake_quant |
| 异常传参场景 | 别名不接受参数且不可调用，无稳定异常路径 | 未覆盖，原因见下 |
| 混合设备类型 | 纯类型别名，不涉及张量或多设备输入 | 不适用 |

未覆盖项及原因：
- 异常传参场景：ObserverOrFakeQuantize 为 TypeAliasType 类型别名，本身不可调用，没有参数校验，
  故无稳定异常路径可用于 assertRaises。
- 空/非空、枚举选项、传参与不传参：类型别名无实例化参数与枚举成员，均不适用。
- 混合设备类型：纯类型别名，不涉及张量或多设备输入，不适用。

注：ObserverOrFakeQuantize 为纯 Python 类型别名，无 CPU/NPU 执行路径，
     故不做 CPU/NPU Golden 对照，仅验证类型别名的联合成员与 isinstance 语义。
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
from torch.ao.quantization import FakeQuantize, MinMaxObserver
from torch.ao.quantization.fake_quantize import FakeQuantizeBase
from torch.ao.quantization.observer import ObserverBase
from typing_extensions import TypeAliasType


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestTorchAoQuantizationObserverOrFakeQuantizeUt(TestCase):
    """Functional tests for the torch.ao.quantization.ObserverOrFakeQuantize type alias."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_alias_is_type_alias_type(self):
        """The public name is a TypeAliasType carrying the documented alias name."""
        alias = torch.ao.quantization.ObserverOrFakeQuantize
        self.assertIsInstance(alias, TypeAliasType)
        self.assertEqual(alias.__name__, "ObserverOrFakeQuantize")

    def test_value_union_members(self):
        """The alias value is the union of ObserverBase and FakeQuantizeBase."""
        value = torch.ao.quantization.ObserverOrFakeQuantize.__value__
        args = value.__args__
        self.assertIn(ObserverBase, args)
        self.assertIn(FakeQuantizeBase, args)

    def test_isinstance_matches_observers_and_fake_quant(self):
        """isinstance against the alias union accepts observer and fake-quant instances."""
        value = torch.ao.quantization.ObserverOrFakeQuantize.__value__
        # A wrong union (e.g. dropping FakeQuantizeBase) would flip this to False.
        self.assertTrue(isinstance(MinMaxObserver(), value))
        self.assertTrue(isinstance(FakeQuantize(), value))

    def test_isinstance_rejects_plain_object(self):
        """isinstance against the alias union rejects objects outside both members."""
        value = torch.ao.quantization.ObserverOrFakeQuantize.__value__
        self.assertFalse(isinstance(3, value))

    def test_alias_module_and_export(self):
        """The alias sets __module__ to the public namespace and is listed in __all__."""
        alias = torch.ao.quantization.ObserverOrFakeQuantize
        self.assertEqual(alias.__module__, "torch.ao.quantization")
        self.assertIn("ObserverOrFakeQuantize", torch.ao.quantization.__all__)


if __name__ == "__main__":
    run_tests()
