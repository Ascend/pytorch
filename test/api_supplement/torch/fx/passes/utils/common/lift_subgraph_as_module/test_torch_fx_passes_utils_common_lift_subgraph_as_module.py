# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.passes.utils.common.lift_subgraph_as_module 接口功能正确性
API 名称：torch.fx.passes.utils.common.lift_subgraph_as_module
API 签名：lift_subgraph_as_module(gm, subgraph, comp_name='', class_name='GraphModule') ->
    tuple[torch.fx.graph_module.GraphModule, dict[str, str]]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空匹配、空节点或无注解状态 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 GraphModule/Node、模块与默认/显式参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：无属性图、默认与显式组件/类名 |
| 正常传参场景 | 已覆盖：嵌套模块/属性提升与 FQN 映射 |
| 异常传参场景 | 未覆盖：无稳定异常契约 |
| 混合设备类型 | 不适用：本批为设备无关的图结构、类型或辅助状态语义 |

未覆盖项及原因：
- 图执行方式：本文件直接验证图模式辅助接口，不声明 NPU Inductor codegen 覆盖。
- eager 对照：使用接口定义的结构、状态或确定值；不构造无意义的 CPU/NPU 自比对。
- autograd：本接口场景不计算梯度。
- RNG：本文件使用确定输入，不依赖随机流。
"""
import os
import unittest

import torch

RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"
SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
DEVICE_NAME = None
if RUN_FLAG:
    try:
        import torch_npu  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "torch_npu":
            raise
    else:
        if torch.npu.is_available():
            DEVICE_NAME = "npu"
    if DEVICE_NAME is None and torch.cuda.is_available():
        DEVICE_NAME = "cuda"

if DEVICE_NAME == "npu":
    from torch_npu.testing.testcase import TestCase, run_tests
else:
    from unittest import TestCase

    def run_tests():
        unittest.main()

import copy
import operator
from torch.fx.passes.utils.common import lift_subgraph_as_module


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_lift_subgraph_as_module_nested_module_and_attribute_are_lifted_with_mapping(self):
        root = torch.nn.Module()
        root.child = torch.nn.Module()
        root.child.identity = torch.nn.Identity()
        root.child.scale = 3
        graph = torch.fx.Graph()
        value = graph.placeholder("value")
        identity = graph.call_module("child.identity", (value,))
        scale = graph.create_node("get_attr", "child.scale", (), {})
        graph.output(graph.call_function(operator.mul, (identity, scale)))
        module = torch.fx.GraphModule(root, graph)
        lifted, mapping = lift_subgraph_as_module(module, copy.deepcopy(module.graph), "piece", "LiftedPiece")
        self.assertIs(lifted.child.identity, module.child.identity)
        self.assertEqual(mapping, {"child.identity": "piece.child.identity", "child.scale": "piece.child.scale"})
        self.assertEqual(lifted(4), 12)
        self.assertEqual(lifted(-2), -6)
        self.assertEqual(type(lifted).__name__, "LiftedPiece")

    def test_lift_subgraph_as_module_attribute_free_graph_uses_default_name_and_empty_mapping(self):
        module = torch.fx.symbolic_trace(lambda value: value + 5)
        lifted, mapping = lift_subgraph_as_module(module, copy.deepcopy(module.graph))
        self.assertEqual(mapping, {})
        self.assertEqual(lifted(0), 5)
        lifted.graph.lint()


if __name__ == "__main__":
    run_tests()
