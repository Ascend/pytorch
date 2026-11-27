# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.fx.utils.NodeInfo.op 接口功能正确性
API 名称：torch.ao.quantization.fx.utils.NodeInfo.op
API 签名：NodeInfo(op, target).op

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | op 取空字符串与正常 op 字符串 | 已覆盖：test_op_can_be_empty_string / test_op_returns_first_field |
| 枚举选项 | FX 节点 op 的有限取值（placeholder/call_method 等） | 已覆盖：test_op_matches_real_graph_node |
| 参数类型 | 构造 NodeInfo 的两个字段均为 str | 已覆盖：test_op_returns_first_field |
| 传参与不传参 | NodeInfo 两位置参数均为必填 | 不适用 |
| 等价类/边界值 | 相同字段相等、不同字段不等 | 已覆盖：test_op_equality_uses_all_fields |
| 正常传参场景 | 由真实 FX 节点构造后读取 op | 已覆盖：test_op_matches_real_graph_node |
| 异常传参场景 | namedtuple 属性访问无稳定异常路径 | 未覆盖，原因见下 |
| 混合设备类型 | 纯 Python 元组字段，无设备输入 | 不适用 |

未覆盖项及原因：
- 异常传参场景：`op` 为 namedtuple 的只读字段，访问不抛异常；字段缺失在构造阶段即被 namedtuple 拒绝，非该字段稳定契约。
- 枚举选项全覆盖：FX 节点 op 取值与后端实现相关，仅覆盖常见取值，未穷举全部字符串。
- 混合设备类型：纯 Python 元组字段，不涉及张量设备，无 CPU/NPU 混合输入路径。

注意：本 API 为纯 Python 数据结构字段（类别 A），无 CPU/NPU 设备执行路径，
     故不做 CPU/NPU Golden 对照，仅按返回值相等性与类型断言。
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

import torch.fx
from torch.ao.quantization.fx.utils import NodeInfo


def _build_size_node():
    graph = torch.fx.Graph()
    inp = graph.placeholder("inp")
    size = graph.call_method("size", (inp,))
    graph.output(size)
    return size


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestNodeInfoOpUt(TestCase):
    """Functional tests for torch.ao.quantization.fx.utils.NodeInfo.op."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_op_returns_first_field(self):
        info = NodeInfo("call_method", "size")
        self.assertEqual(info.op, "call_method")

    def test_op_is_namedtuple_field(self):
        self.assertEqual(NodeInfo._fields, ("op", "target"))
        info = NodeInfo("call_method", "size")
        self.assertEqual(info._asdict()["op"], "call_method")
        self.assertEqual(info[0], "call_method")

    def test_op_matches_real_graph_node(self):
        node = _build_size_node()
        info = NodeInfo(node.op, node.target)
        self.assertEqual(info.op, node.op)
        self.assertEqual(info.op, "call_method")
        self.assertEqual(info.target, "size")

    def test_op_can_be_empty_string(self):
        info = NodeInfo("", None)
        self.assertEqual(info.op, "")

    def test_op_equality_uses_all_fields(self):
        self.assertEqual(NodeInfo("call_method", "size"), NodeInfo("call_method", "size"))
        self.assertNotEqual(NodeInfo("call_method", "size"), NodeInfo("call_method", "view"))

    def test_op_reflects_value_not_target(self):
        info = NodeInfo("placeholder", "call_method")
        self.assertEqual(info.op, "placeholder")


if __name__ == "__main__":
    run_tests()
