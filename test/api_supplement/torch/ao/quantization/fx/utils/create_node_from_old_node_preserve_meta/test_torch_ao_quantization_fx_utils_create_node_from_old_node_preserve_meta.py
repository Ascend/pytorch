# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.fx.utils.create_node_from_old_node_preserve_meta 接口功能正确性
API 名称：torch.ao.quantization.fx.utils.create_node_from_old_node_preserve_meta
API 签名：create_node_from_old_node_preserve_meta(quantized_graph, create_node_args, old_node) -> Node

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | old_node.stack_trace 为空字符串与有效字符串 | 已覆盖：test_copies_empty_stack_trace / test_copies_stack_trace |
| 枚举选项 | 该 API 无枚举参数 | 不适用 |
| 参数类型 | quantized_graph 为 Graph、create_node_args 为 tuple | 已覆盖：test_creates_node_with_given_op_and_target |
| 传参与不传参 | 三参数均为必填 | 不适用 |
| 等价类/边界值 | 同一 meta 复制到不同新节点 | 已覆盖：test_copies_stack_trace |
| 正常传参场景 | 在新图上建节点并复制 stack_trace | 已覆盖：test_creates_node_with_given_op_and_target |
| 异常传参场景 | 无非稳定异常路径 | 未覆盖，原因见下 |
| 混合设备类型 | 纯图结构工具，无设备输入 | 不适用 |

未覆盖项及原因：
- 异常传参场景：函数仅调用 graph.create_node 并复制 stack_trace，非法参数由底层 FX 抛出，
  非本 API 稳定契约，故不单列。
- 枚举选项：该 API 无枚举参数。
- 混合设备类型：纯图结构工具，不涉及张量设备，无 CPU/NPU 混合输入路径。

注意：本 API 为纯 Python 图工具（类别 A），无 CPU/NPU 设备执行路径，
     故不做 CPU/NPU Golden 对照，仅按返回节点的 op/target 与 stack_trace 断言。
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
from torch.ao.quantization.fx.utils import create_node_from_old_node_preserve_meta


def _build_old_node(stack_trace):
    graph = torch.fx.Graph()
    inp = graph.placeholder("inp")
    old = graph.call_function(torch.neg, (inp,))
    old.stack_trace = stack_trace
    graph.output(old)
    return old


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestCreateNodeFromOldNodePreserveMetaUt(TestCase):
    """Functional tests for torch.ao.quantization.fx.utils.create_node_from_old_node_preserve_meta."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_creates_node_with_given_op_and_target(self):
        old = _build_old_node("trace-neg")
        graph = torch.fx.Graph()
        inp = graph.placeholder("inp")
        new_node = create_node_from_old_node_preserve_meta(graph, ("call_function", torch.relu, (inp,)), old)
        self.assertIsInstance(new_node, torch.fx.Node)
        self.assertEqual(new_node.op, "call_function")
        self.assertIs(new_node.target, torch.relu)

    def test_copies_stack_trace(self):
        old = _build_old_node("trace-neg")
        graph = torch.fx.Graph()
        new_node = create_node_from_old_node_preserve_meta(graph, ("call_function", torch.relu, ()), old)
        self.assertEqual(new_node.stack_trace, "trace-neg")
        self.assertEqual(new_node.stack_trace, old.stack_trace)

    def test_copies_empty_stack_trace(self):
        old = _build_old_node("")
        graph = torch.fx.Graph()
        new_node = create_node_from_old_node_preserve_meta(graph, ("call_function", torch.relu, ()), old)
        self.assertEqual(new_node.stack_trace, "")

    def test_node_added_to_graph(self):
        old = _build_old_node("trace-neg")
        graph = torch.fx.Graph()
        new_node = create_node_from_old_node_preserve_meta(graph, ("call_function", torch.relu, ()), old)
        self.assertIn(new_node, list(graph.nodes))

    def test_old_node_is_not_modified(self):
        old = _build_old_node("trace-neg")
        graph = torch.fx.Graph()
        create_node_from_old_node_preserve_meta(graph, ("call_function", torch.relu, ()), old)
        self.assertEqual(old.stack_trace, "trace-neg")
        self.assertIs(old.target, torch.neg)


if __name__ == "__main__":
    run_tests()
