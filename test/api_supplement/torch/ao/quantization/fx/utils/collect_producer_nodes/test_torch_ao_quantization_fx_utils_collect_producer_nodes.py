# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.fx.utils.collect_producer_nodes 接口功能正确性
API 名称：torch.ao.quantization.fx.utils.collect_producer_nodes
API 签名：collect_producer_nodes(node) -> list[Node] | None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | 起始节点无生产者（自包含）与有生产者两种 | 已覆盖：test_single_get_attr_node_returns_itself / test_collects_chain_from_observer |
| 枚举选项 | 链中节点 op：call_module/get_attr/placeholder | 已覆盖：test_collect_includes_get_attr 等 |
| 参数类型 | 入参为 FX Node | 已覆盖：test_collects_chain_from_observer |
| 传参与不传参 | node 为唯一必填参数 | 不适用 |
| 等价类/边界值 | 命中 placeholder（返回 None）为边界 | 已覆盖：test_linear_node_returns_none |
| 正常传参场景 | 从 call_module 节点回溯出生产者链 | 已覆盖：test_collects_chain_from_observer |
| 异常传参场景 | 无自由变量时返回 None（非异常） | 已覆盖：test_linear_node_returns_none |
| 混合设备类型 | 纯图结构遍历，无设备输入 | 不适用 |

未覆盖项及原因：
- 异常传参场景：函数以 None 表达「无法提取自包含子图」，不抛异常，故无 assertRaises 路径。
- 枚举选项全覆盖：链中节点类型随量化图而变，仅覆盖常见的 call_module/get_attr/placeholder。
- 混合设备类型：纯图结构遍历，不涉及张量设备，无 CPU/NPU 混合输入路径。

注意：本 API 为纯 Python 图结构工具（类别 A），无 CPU/NPU 设备执行路径，
     故不做 CPU/NPU Golden 对照，仅按返回列表的节点内容/顺序与 None 边界断言。
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
from torch.ao.quantization.fx.utils import collect_producer_nodes


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
    return gm, observed, linear, weight


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestCollectProducerNodesUt(TestCase):
    """Functional tests for torch.ao.quantization.fx.utils.collect_producer_nodes."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_collects_chain_from_observer(self):
        _, observed, _, _ = _build_observer_graph()
        nodes = collect_producer_nodes(observed)
        self.assertIsInstance(nodes, list)
        self.assertEqual(nodes[0], observed)
        self.assertEqual(len(nodes), 2)

    def test_collect_includes_get_attr(self):
        _, observed, _, weight = _build_observer_graph()
        nodes = collect_producer_nodes(observed)
        self.assertIn(weight, nodes)
        self.assertEqual([n.op for n in nodes], ["call_module", "get_attr"])

    def test_linear_node_returns_none(self):
        _, _, linear, _ = _build_observer_graph()
        # The linear node consumes the placeholder input, so no self-contained
        # subgraph can be extracted.
        self.assertIsNone(collect_producer_nodes(linear))

    def test_single_get_attr_node_returns_itself(self):
        _, _, _, weight = _build_observer_graph()
        nodes = collect_producer_nodes(weight)
        self.assertEqual(nodes, [weight])

    def test_result_preserves_backward_order(self):
        _, observed, _, weight = _build_observer_graph()
        nodes = collect_producer_nodes(observed)
        self.assertEqual(nodes[-1], weight)


if __name__ == "__main__":
    run_tests()
