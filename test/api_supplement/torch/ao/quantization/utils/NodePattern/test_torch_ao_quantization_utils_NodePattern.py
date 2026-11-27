# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.utils.NodePattern 接口功能正确性
API 名称：torch.ao.quantization.utils.NodePattern
API 签名：NodePattern = TypeAliasType("NodePattern",
          tuple[Node, Node] | tuple[Node, tuple[Node, Node]] | Any)

覆盖维度表：
| 覆盖维度      | 说明                                  | 覆盖情况                                                 |
|---------------|---------------------------------------|----------------------------------------------------------|
| 空/非空       | 别名定义非空（__value__ 非 None）     | 已覆盖：test_node_pattern_value_is_union                 |
| 枚举选项      | 该别名无枚举参数                      | 不适用                                                   |
| 参数类型      | 别名底层 union 聚合 tuple[Node,...]   | 已覆盖：test_node_pattern_value_is_union                 |
|               | 与 Any                                |                                                          |
| 传参与不传参  | 该别名无可选参数                      | 不适用                                                   |
| 等价类/边界值 | 同名不同实例、别名与自身相等/同一     | 已覆盖：test_node_pattern_identity_equality、            |
|               |                                       | test_node_pattern_distinct_alias_not_equal               |
| 正常传参场景  | 别名可哈希、可作字典键、              | 已覆盖：test_node_pattern_hashable、                     |
|               | 非泛型不可下标                        | test_node_pattern_not_generic                            |
| 异常传参场景  | 非泛型别名下标访问触发 TypeError      | 已覆盖：test_node_pattern_not_generic                    |
| 混合设备类型  | 多 Tensor 异构设备输入                | 不适用，该别名仅为类型对象，不涉及设备                   |

未覆盖项及原因：
- 枚举选项、传参与不传参：类型别名无运行期参数，无对应维度可覆盖。
- 字段级（dataclass）语义：本版本 NodePattern 由 typing_extensions.TypeAliasType 定义，并非 dataclass，
  无 dataclass 字段，故改以别名的 __name__/__value__/__eq__/__hash__ 语义验证。
- 混合设备类型：纯类型对象，无 Tensor 执行路径，故不覆盖。

注：该 API 为纯 Python 类型对象（框架类 A），无 CPU/NPU 数值输出，不做 CPU Golden 对照。
"""
import os
import typing
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

from typing_extensions import TypeAliasType

import torch.ao.quantization.utils as quant_utils
from torch.ao.quantization.utils import NodePattern


def _node_leaves(value):
    """Flatten the union arguments, expanding nested tuples, to reach Node leaves."""
    leaves = []
    for arg in typing.get_args(value):
        if typing.get_origin(arg) is tuple:
            for inner in typing.get_args(arg):
                if typing.get_origin(inner) is tuple:
                    leaves.extend(typing.get_args(inner))
                else:
                    leaves.append(inner)
    return leaves


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestNodePatternUt(TestCase):
    """Functional tests for torch.ao.quantization.utils.NodePattern."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_node_pattern_is_type_alias(self):
        self.assertIsInstance(NodePattern, TypeAliasType)
        self.assertIn("NodePattern", quant_utils.__all__)

    def test_node_pattern_value_is_union(self):
        value = NodePattern.__value__
        self.assertIsNotNone(value)
        self.assertEqual(NodePattern.__name__, "NodePattern")
        args = typing.get_args(value)
        self.assertIn(typing.Any, args)
        self.assertIn(torch.fx.Node, _node_leaves(value))

    def test_node_pattern_identity_equality(self):
        # Equality is identity based: the alias equals itself.
        self.assertTrue(NodePattern == NodePattern)
        self.assertTrue(NodePattern is NodePattern)

    def test_node_pattern_distinct_alias_not_equal(self):
        other = TypeAliasType("NodePattern", int)
        self.assertFalse(NodePattern == other)
        self.assertFalse(NodePattern is other)

    def test_node_pattern_hashable(self):
        mapping = {NodePattern: 1}
        self.assertEqual(mapping[NodePattern], 1)
        other = TypeAliasType("NodePattern", int)
        mapping[other] = 2
        # Distinct instances stay distinct even with the same name.
        self.assertEqual(len(mapping), 2)

    def test_node_pattern_not_generic(self):
        with self.assertRaises(TypeError):
            NodePattern[torch.fx.Node]


if __name__ == "__main__":
    run_tests()
