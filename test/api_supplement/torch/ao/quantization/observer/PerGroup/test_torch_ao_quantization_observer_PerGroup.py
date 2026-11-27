# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.observer.PerGroup 数据类接口功能正确性
API 名称：torch.ao.quantization.observer.PerGroup
API 签名：PerGroup(group_size: int)  # frozen dataclass，继承 Granularity

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | frozen dataclass 只有必填 group_size 字段，无空态 | 不适用 |
| 枚举选项 | 该数据类无枚举参数 | 不适用 |
| 参数类型 | group_size 为 int，驱动分组块大小 | 已覆盖：test_per_group_stores_group_size |
| 传参与不传参 | group_size 必填，不传参触发 TypeError | 已覆盖：test_per_group_missing_argument_raises |
| 等价类/边界值 | group_size=1（最小）、group_size=16（常规） | 已覆盖：test_per_group_equality_and_hash |
| 正常传参场景 | 构造 PerGroup 并作为 Granularity 求 block_size | 已覆盖：test_get_block_size_for_per_group |
| 异常传参场景 | 缺少必填参数触发 TypeError | 已覆盖：test_per_group_missing_argument_raises |
| 混合设备类型 | 纯 Python 数据类，不涉及张量或多设备输入 | 不适用 |

未覆盖项及原因：
- 空/非空：PerGroup 为 frozen dataclass，仅有单一必填 group_size 字段，没有 None 与合法值的语义差异。
- 枚举选项、混合设备类型：数据类无枚举参数、不涉及张量设备，均不适用。
- 冻结校验：PerGroup 为 frozen dataclass，字段赋值抛 FrozenInstanceError，见
  test_per_group_is_frozen。

注：PerGroup 为纯 Python 数据类，无 CPU/NPU 执行路径，
     故不做 CPU/NPU Golden 对照，仅验证字段、冻结语义与 block_size 计算。
"""
import os
import unittest
from dataclasses import FrozenInstanceError, fields

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

from torch.ao.quantization.observer import Granularity, PerGroup, get_block_size


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestTorchAoQuantizationObserverPerGroupUt(TestCase):
    """Functional tests for torch.ao.quantization.observer.PerGroup."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_per_group_stores_group_size(self):
        """Construction stores the exact int group_size and exposes it as the only field."""
        group = PerGroup(4)
        self.assertEqual(group.group_size, 4)
        self.assertEqual([field.name for field in fields(PerGroup)], ["group_size"])

    def test_per_group_is_granularity(self):
        """PerGroup is a Granularity subclass so granularity dispatch works."""
        self.assertIsInstance(PerGroup(4), Granularity)

    def test_per_group_is_frozen(self):
        """A frozen dataclass rejects in-place field mutation."""
        group = PerGroup(4)
        with self.assertRaises(FrozenInstanceError):
            group.group_size = 8

    def test_per_group_equality_and_hash(self):
        """Equal group_size yields equal/hashable instances; different values differ."""
        self.assertEqual(PerGroup(4), PerGroup(4))
        self.assertNotEqual(PerGroup(4), PerGroup(8))
        # group_size=1 is the minimal legal group; it must hash and be distinct from 16.
        self.assertNotEqual(PerGroup(1), PerGroup(16))
        self.assertEqual(hash(PerGroup(16)), hash(PerGroup(16)))

    def test_get_block_size_for_per_group(self):
        """A 2-D input with PerGroup(n) yields a (1, n) block size."""
        self.assertEqual(get_block_size((8, 16), PerGroup(4)), (1, 4))
        self.assertEqual(get_block_size((8, 16), PerGroup(16)), (1, 16))

    def test_per_group_missing_argument_raises(self):
        """Omitting the required group_size is rejected by the dataclass constructor."""
        with self.assertRaises(TypeError):
            PerGroup()

    def test_per_group_repr_contains_group_size(self):
        """The dataclass repr mirrors the field so debugging output is meaningful."""
        self.assertIn("group_size=4", repr(PerGroup(4)))


if __name__ == "__main__":
    run_tests()
