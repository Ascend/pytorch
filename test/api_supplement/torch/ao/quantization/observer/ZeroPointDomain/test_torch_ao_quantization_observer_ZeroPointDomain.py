# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.observer.ZeroPointDomain 枚举接口功能正确性
API 名称：torch.ao.quantization.observer.ZeroPointDomain
API 签名：class ZeroPointDomain(Enum): INT = auto(); FLOAT = auto(); NONE = auto()

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | 枚举类无实例化输入，无空/非空语义 | 不适用 |
| 枚举选项 | 全部成员 INT/FLOAT/NONE | 已覆盖：test_all_members_present / test_auto_values_are_sequential |
| 参数类型 | 枚举类无参数字段，仅按 name/value 访问 | 已覆盖：test_lookup_by_name_and_value |
| 传参与不传参 | 枚举构造不可传参 | 不适用 |
| 等价类/边界值 | 首成员/末成员及 name/value 双向查找 | 已覆盖：test_auto_values_are_sequential / test_lookup_by_name_and_value |
| 正常传参场景 | 迭代枚举并获得全部成员 | 已覆盖：test_iteration_order_and_length |
| 异常传参场景 | 非法成员名 KeyError、非法值 ValueError | 已覆盖：test_invalid_member_raises |
| 混合设备类型 | 纯 Python 枚举，不涉及张量或多设备输入 | 不适用 |

未覆盖项及原因：
- 空/非空、传参与不传参、混合设备类型：ZeroPointDomain 为纯 Enum 类，无实例化输入、
  无构造参数、不涉及张量设备，均不适用。

注：ZeroPointDomain 为纯 Python 枚举，无 CPU/NPU 执行路径，
     故不做 CPU/NPU Golden 对照，仅验证成员集合、自动取值与查找/异常语义。
"""
import os
import unittest
from enum import Enum

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

from torch.ao.quantization.observer import ZeroPointDomain


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestTorchAoQuantizationObserverZeroPointDomainUt(TestCase):
    """Functional tests for torch.ao.quantization.observer.ZeroPointDomain."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_is_enum_subclass(self):
        """ZeroPointDomain derives from Enum so it supports name/value lookups."""
        self.assertTrue(issubclass(ZeroPointDomain, Enum))

    def test_all_members_present(self):
        """The three documented members exist by exact name."""
        for name in ("INT", "FLOAT", "NONE"):
            self.assertIn(name, ZeroPointDomain.__members__)
        self.assertIs(ZeroPointDomain.INT, ZeroPointDomain["INT"])
        self.assertIs(ZeroPointDomain.FLOAT, ZeroPointDomain["FLOAT"])
        self.assertIs(ZeroPointDomain.NONE, ZeroPointDomain["NONE"])

    def test_auto_values_are_sequential(self):
        """auto() assigns sequential 1/2/3 values with distinct identities."""
        self.assertEqual(ZeroPointDomain.INT.value, 1)
        self.assertEqual(ZeroPointDomain.FLOAT.value, 2)
        self.assertEqual(ZeroPointDomain.NONE.value, 3)
        # A duplicated auto value would collapse members and break this identity check.
        self.assertEqual(len(list(ZeroPointDomain)), 3)

    def test_lookup_by_name_and_value(self):
        """Both name and value based lookups return the same canonical members."""
        self.assertIs(ZeroPointDomain["FLOAT"], ZeroPointDomain.FLOAT)
        self.assertIs(ZeroPointDomain(1), ZeroPointDomain.INT)
        self.assertIs(ZeroPointDomain(3), ZeroPointDomain.NONE)

    def test_iteration_order_and_length(self):
        """Iteration yields members in definition order for the three options."""
        self.assertEqual([member.name for member in ZeroPointDomain], ["INT", "FLOAT", "NONE"])

    def test_invalid_member_raises(self):
        """Unknown names/values raise KeyError/ValueError respectively."""
        with self.assertRaises(KeyError):
            ZeroPointDomain["MISSING"]
        with self.assertRaises(ValueError):
            ZeroPointDomain(99)


if __name__ == "__main__":
    run_tests()
