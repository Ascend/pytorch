# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.utils.Pattern 接口功能正确性
API 名称：torch.ao.quantization.utils.Pattern
API 签名：Pattern = TypeAliasType("Pattern",
          Callable | tuple[Callable, Callable] | tuple[Callable, tuple[Callable, Callable]] | Any)

覆盖维度表：
| 覆盖维度      | 说明                                         | 覆盖情况                                          |
|---------------|----------------------------------------------|---------------------------------------------------|
| 空/非空       | 别名定义非空（__value__ 非 None）            | 已覆盖：test_pattern_value_is_union               |
| 枚举选项      | 该别名无枚举参数                             | 不适用                                            |
| 参数类型      | 别名底层 union 聚合 Callable/tuple 与 Any    | 已覆盖：test_pattern_value_is_union               |
| 传参与不传参  | 该别名无可选参数                             | 不适用                                            |
| 等价类/边界值 | 同名不同实例、与自身相等/同一、与            | 已覆盖：test_pattern_identity_equality、          |
|               | NodePattern 不同                             |                                                   |
|               |                                              | test_pattern_distinct_alias_not_equal             |
| 正常传参场景  | 别名可哈希、可作字典键、非泛型不可下标       | 已覆盖：test_pattern_hashable、                   |
|               |                                              | test_pattern_not_generic                          |
| 异常传参场景  | 非泛型别名下标访问触发 TypeError             | 已覆盖：test_pattern_not_generic                  |
| 混合设备类型  | 多 Tensor 异构设备输入                       | 不适用，该别名仅为类型对象，不涉及设备            |

未覆盖项及原因：
- 枚举选项、传参与不传参：类型别名无运行期参数，无对应维度可覆盖。
- 字段级（dataclass）语义：本版本 Pattern 由 typing_extensions.TypeAliasType 定义，并非 dataclass，
  无 dataclass 字段，故改以别名的 __name__/__value__/__eq__/__hash__ 语义验证。
- 混合设备类型：纯类型对象，无 Tensor 执行路径，故不覆盖。

注：该 API 为纯 Python 类型对象（框架类 A），无 CPU/NPU 数值输出，不做 CPU Golden 对照。
"""
import os
import typing
import unittest
from collections.abc import Callable

import torch  # noqa: F401  (torch is imported for parity with the shared probe block)

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

from typing_extensions import TypeAliasType

import torch.ao.quantization.utils as quant_utils
from torch.ao.quantization.utils import NodePattern, Pattern


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestPatternUt(TestCase):
    """Functional tests for torch.ao.quantization.utils.Pattern."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_pattern_is_type_alias(self):
        self.assertIsInstance(Pattern, TypeAliasType)
        self.assertIn("Pattern", quant_utils.__all__)

    def test_pattern_value_is_union(self):
        value = Pattern.__value__
        self.assertIsNotNone(value)
        self.assertEqual(Pattern.__name__, "Pattern")
        args = typing.get_args(value)
        self.assertIn(typing.Any, args)
        self.assertIn(Callable, args)

    def test_pattern_identity_equality(self):
        self.assertTrue(Pattern == Pattern)
        self.assertTrue(Pattern is Pattern)

    def test_pattern_distinct_alias_not_equal(self):
        other = TypeAliasType("Pattern", int)
        self.assertFalse(Pattern == other)
        self.assertFalse(Pattern is other)

    def test_pattern_differs_from_node_pattern(self):
        self.assertFalse(Pattern == NodePattern)
        self.assertFalse(Pattern is NodePattern)

    def test_pattern_hashable(self):
        mapping = {Pattern: 1}
        self.assertEqual(mapping[Pattern], 1)
        other = TypeAliasType("Pattern", int)
        mapping[other] = 2
        self.assertEqual(len(mapping), 2)

    def test_pattern_not_generic(self):
        with self.assertRaises(TypeError):
            Pattern[Callable]


if __name__ == "__main__":
    run_tests()
