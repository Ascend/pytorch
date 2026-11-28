# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.passes.utils.fuser_utils.fuse_as_graphmodule 接口功能正确性
API 名称：torch.fx.passes.utils.fuser_utils.fuse_as_graphmodule
API 签名：fuse_as_graphmodule(gm, nodes, module_name, partition_lookup_table=None, *, always_return_tuple=False) ->
    tuple[torch.fx.graph_module.GraphModule, tuple[torch.fx.node.Node, ...], tuple[torch.fx.node.Node, ...]]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空匹配、空节点或无注解状态 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 GraphModule/Node、模块与默认/显式参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：显式查找表、tuple 输出及外部节点拒绝 |
| 正常传参场景 | 已覆盖：融合图执行、输入顺序和原图保持 |
| 异常传参场景 | 已覆盖：不属于目标图的节点 AssertionError |
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
from torch.fx.passes.utils.fuser_utils import fuse_as_graphmodule


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_fuse_as_graphmodule_fusion_keeps_input_order_outputs_and_original_graph(self):
        module = torch.fx.symbolic_trace(lambda x, y: y * 2 + x)
        original = tuple(module.graph.nodes)
        inputs = tuple(node for node in original if node.op == "placeholder")
        nodes = [node for node in original if node.op == "call_function"]
        fused, actual_inputs, outputs = fuse_as_graphmodule(module, nodes, "FusedPiece")
        self.assertEqual(actual_inputs, inputs)
        self.assertEqual(outputs, (nodes[-1],))
        self.assertEqual(fused(3, 4), 11)
        self.assertEqual(fused(-2, 5), 8)
        self.assertEqual(tuple(module.graph.nodes), original)
        fused.graph.lint()

    def test_fuse_as_graphmodule_tuple_option_and_foreign_node_rejection(self):
        module = torch.fx.symbolic_trace(lambda x: x + 2)
        nodes = [node for node in module.graph.nodes if node.op == "call_function"]
        fused, _, _ = fuse_as_graphmodule(module, nodes, "TuplePiece", dict.fromkeys(nodes), always_return_tuple=True)
        self.assertEqual(fused(3), (5,))
        foreign = torch.fx.symbolic_trace(lambda x: -x)
        other = next(node for node in foreign.graph.nodes if node.target is operator.neg)
        with self.assertRaisesRegex(AssertionError, "doesn't belong"):
            fuse_as_graphmodule(module, [other], "InvalidPiece")


if __name__ == "__main__":
    run_tests()
