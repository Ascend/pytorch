# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.fx.utils.graph_module_from_producer_nodes 接口功能正确性
API 名称：torch.ao.quantization.fx.utils.graph_module_from_producer_nodes
API 签名：graph_module_from_producer_nodes(root, producer_nodes) -> GraphModule

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | producer_nodes 为空的边界与正常非空列表 | 已覆盖：test_empty_producer_nodes_raises / test_builds_graph_module |
| 枚举选项 | 该 API 无枚举参数 | 不适用 |
| 参数类型 | root 为 GraphModule、producer_nodes 为 list[Node] | 已覆盖：test_returns_graph_module_instance |
| 传参与不传参 | 两参数均为必填 | 不适用 |
| 等价类/边界值 | 单个生产者节点与多节点链 | 已覆盖：test_builds_graph_module |
| 正常传参场景 | 由生产者节点构造含 output 的 GraphModule | 已覆盖：test_output_is_last_producer_node |
| 异常传参场景 | 空列表抛 AssertionError | 已覆盖：test_empty_producer_nodes_raises |
| 混合设备类型 | 纯图结构构造，无设备输入 | 不适用 |

未覆盖项及原因：
- 枚举选项 / 传参与不传参：该 API 无枚举参数、无可选参数。
- 混合设备类型：纯图结构构造工具，不涉及张量设备，无 CPU/NPU 混合输入路径。

注意：本 API 为纯 Python 图工具（类别 A），无 CPU/NPU 设备执行路径，
     故不做 CPU/NPU Golden 对照，仅按返回 GraphModule 的图结构与节点顺序断言。
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
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization.fx.utils import (
    collect_producer_nodes,
    graph_module_from_producer_nodes,
)


class _Identity(nn.Module):
    def forward(self, x):
        return x


def _build_observer_graph():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    weight = graph.get_attr("weight")
    observed = graph.call_module("observer", (weight,))
    linear = graph.call_function(F.linear, (x, observed))
    graph.output(linear)
    root = nn.Module()
    root.weight = nn.Parameter(torch.randn(3, 3))
    root.observer = _Identity()
    gm = torch.fx.GraphModule(root, graph)
    return gm, observed, weight


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestGraphModuleFromProducerNodesUt(TestCase):
    """Functional tests for graph_module_from_producer_nodes."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_builds_graph_module(self):
        gm, observed, _ = _build_observer_graph()
        producer_nodes = collect_producer_nodes(observed)
        result = graph_module_from_producer_nodes(gm, producer_nodes)
        self.assertIsInstance(result, torch.fx.GraphModule)

    def test_returns_graph_module_instance(self):
        gm, observed, _ = _build_observer_graph()
        producer_nodes = collect_producer_nodes(observed)
        result = graph_module_from_producer_nodes(gm, producer_nodes)
        self.assertIsInstance(result, torch.fx.GraphModule)

    def test_reversed_node_order(self):
        gm, observed, _ = _build_observer_graph()
        producer_nodes = collect_producer_nodes(observed)
        result = graph_module_from_producer_nodes(gm, producer_nodes)
        ops = [node.op for node in result.graph.nodes]
        # Producer nodes are traced backwards, so the copied graph must start
        # from the get_attr node and end with the observer call_module.
        self.assertEqual(ops, ["get_attr", "call_module", "output"])

    def test_output_is_last_producer_node(self):
        gm, observed, _ = _build_observer_graph()
        producer_nodes = collect_producer_nodes(observed)
        result = graph_module_from_producer_nodes(gm, producer_nodes)
        out_node = [n for n in result.graph.nodes if n.op == "output"][0]
        self.assertEqual(out_node.args[0].op, "call_module")
        self.assertEqual(out_node.args[0].target, "observer")

    def test_copied_nodes_do_not_alias_originals(self):
        gm, observed, _ = _build_observer_graph()
        producer_nodes = collect_producer_nodes(observed)
        result = graph_module_from_producer_nodes(gm, producer_nodes)
        copied = [n for n in result.graph.nodes if n.op == "call_module"][0]
        self.assertIsNot(copied, observed)

    def test_empty_producer_nodes_raises(self):
        gm, _, _ = _build_observer_graph()
        with self.assertRaises(AssertionError):
            graph_module_from_producer_nodes(gm, [])


if __name__ == "__main__":
    run_tests()
