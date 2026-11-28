# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.optimization.extract_subgraph 接口功能正确性
API 名称：torch.fx.experimental.optimization.extract_subgraph
API 签名：extract_subgraph(orig_module, nodes, inputs, outputs) -> torch.fx.graph_module.GraphModule

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空节点集合或无匹配状态 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 FX 图、模块、schema 及接口默认/显式参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：中间节点作为边界输入和空输出 |
| 正常传参场景 | 已覆盖：子图提取、多个有序输出及可执行语义 |
| 异常传参场景 | 未覆盖：没有稳定的异常契约 |
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

import operator
from torch.fx.experimental.optimization import extract_subgraph


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_extract_subgraph_multiple_outputs_and_original_graph_are_preserved(self):
        module = torch.fx.symbolic_trace(lambda x, y: x * 2 + y)
        original = tuple(module.graph.nodes)
        x, y = [node for node in original if node.op == "placeholder"]
        multiply = next(node for node in original if node.target is operator.mul)
        add = next(node for node in original if node.target is operator.add)
        result = extract_subgraph(module, [multiply, add], [x, y], [multiply, add])
        self.assertEqual(result(3, 4), [6, 10])
        self.assertEqual(result(-2, 5), [-4, 1])
        self.assertEqual(tuple(module.graph.nodes), original)
        result.graph.lint()

    def test_extract_subgraph_boundary_value_can_replace_a_producer_and_empty_outputs(self):
        module = torch.fx.symbolic_trace(lambda x, y: x * 2 + y)
        x, y = [node for node in module.graph.nodes if node.op == "placeholder"]
        multiply = next(node for node in module.graph.nodes if node.target is operator.mul)
        add = next(node for node in module.graph.nodes if node.target is operator.add)
        result = extract_subgraph(module, [add], [multiply, y], [add])
        self.assertEqual(result(8, 4), [12])
        self.assertEqual(extract_subgraph(module, [], [x], [])(7), [])


if __name__ == "__main__":
    run_tests()
