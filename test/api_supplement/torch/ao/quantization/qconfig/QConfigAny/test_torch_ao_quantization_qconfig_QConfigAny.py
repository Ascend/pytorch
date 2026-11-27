# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.qconfig.QConfigAny 类型别名接口正确性
API 名称：torch.ao.quantization.qconfig.QConfigAny
API 签名：QConfigAny = TypeAliasType("QConfigAny", QConfig | None)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | 别名联合显式允许 None（空）与 QConfig（非空） | 已覆盖：test_none_matches_union / test_qconfig_instance_matches_union |
| 枚举选项 | 该类型别名无枚举参数 | 不适用 |
| 参数类型 | 联合成员为 QConfig 与 NoneType | 已覆盖：test_value_union_members |
| 传参与不传参 | 类型别名无构造参数 | 不适用 |
| 等价类/边界值 | 别名可作为 isinstance 判据容纳 QConfig 与 None | 已覆盖：test_qconfig_instance_matches_union / test_none_matches_union |
| 正常传参场景 | QConfig 实例属于该别名 | 已覆盖：test_qconfig_instance_matches_union |
| 异常传参场景 | 别名不接受参数且不可调用，无稳定异常路径 | 未覆盖，原因见下 |
| 混合设备类型 | 纯类型别名，不涉及张量或多设备输入 | 不适用 |

未覆盖项及原因：
- 异常传参场景：QConfigAny 为 TypeAliasType 类型别名，本身不可调用，没有参数校验，
  故无稳定异常路径可用于 assertRaises。
- 枚举选项、传参与不传参、混合设备类型：类型别名无枚举成员、无构造参数、不涉及张量设备，均不适用。

注：QConfigAny 为纯 Python 类型别名，无 CPU/NPU 执行路径，
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
import torch.ao.quantization.qconfig
from torch.ao.quantization.observer import (
    default_observer,
    default_weight_observer,
)
from torch.ao.quantization.qconfig import QConfig, QConfigAny
from typing_extensions import TypeAliasType


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestTorchAoQuantizationQconfigQConfigAnyUt(TestCase):
    """Functional tests for the torch.ao.quantization.qconfig.QConfigAny type alias."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_alias_is_type_alias_type(self):
        """The public name is a TypeAliasType carrying the documented alias name."""
        self.assertIsInstance(QConfigAny, TypeAliasType)
        self.assertEqual(QConfigAny.__name__, "QConfigAny")

    def test_value_union_members(self):
        """The alias value is the union of QConfig and NoneType."""
        args = QConfigAny.__value__.__args__
        self.assertIn(QConfig, args)
        self.assertIn(type(None), args)

    def test_qconfig_instance_matches_union(self):
        """A real QConfig instance is accepted by the alias union."""
        qconfig = QConfig(activation=default_observer, weight=default_weight_observer)
        self.assertTrue(isinstance(qconfig, QConfigAny.__value__))

    def test_none_matches_union(self):
        """None is explicitly part of the union so unquantized layers type-check."""
        self.assertTrue(isinstance(None, QConfigAny.__value__))

    def test_plain_object_rejected(self):
        """Objects outside QConfig/None are rejected by the union."""
        self.assertFalse(isinstance(3, QConfigAny.__value__))

    def test_alias_module_and_export(self):
        """The alias lives in the qconfig module and is exported in __all__."""
        self.assertEqual(QConfigAny.__module__, "torch.ao.quantization.qconfig")
        self.assertIn("QConfigAny", torch.ao.quantization.qconfig.__all__)
        self.assertIn("QConfigAny", torch.ao.quantization.__all__)


if __name__ == "__main__":
    run_tests()
