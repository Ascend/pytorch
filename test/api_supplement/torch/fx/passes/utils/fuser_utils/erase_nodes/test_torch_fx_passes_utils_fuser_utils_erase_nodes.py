# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.passes.utils.fuser_utils.erase_nodes 接口功能正确性
API 名称：torch.fx.passes.utils.fuser_utils.erase_nodes
API 签名：erase_nodes(gm, nodes) -> None

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空匹配、空节点或无注解状态 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 GraphModule/Node、模块与默认/显式参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：空删除及仍被使用节点拒绝 |
| 正常传参场景 | 已覆盖：依赖链逆拓扑删除及剩余图执行 |
| 异常传参场景 | 已覆盖：删除活跃节点的 RuntimeError |
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
from torch.fx.passes.utils.fuser_utils import erase_nodes


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_erase_nodes_dead_dependency_chain_is_erased_in_reverse_order(self):
        module = torch.fx.symbolic_trace(lambda value: value * 2 + 1)
        source = next(node for node in module.graph.nodes if node.op == "placeholder")
        output = next(node for node in module.graph.nodes if node.op == "output")
        with module.graph.inserting_before(output):
            first = module.graph.call_function(operator.neg, (source,))
            last = module.graph.call_function(operator.abs, (first,))
        erase_nodes(module, [first, last])
        self.assertNotIn(first, list(module.graph.nodes))
        self.assertNotIn(last, list(module.graph.nodes))
        module.graph.lint()
        module.recompile()
        self.assertEqual(module(3), 7)

    def test_erase_nodes_empty_erasure_and_live_dependency_rejection(self):
        module = torch.fx.symbolic_trace(lambda value: value + 2)
        before = tuple(module.graph.nodes)
        self.assertIsNone(erase_nodes(module, []))
        self.assertEqual(tuple(module.graph.nodes), before)
        live = next(node for node in module.graph.nodes if node.op == "call_function")
        with self.assertRaises(RuntimeError):
            erase_nodes(module, [live])
        self.assertEqual(module(1), 3)


if __name__ == "__main__":
    run_tests()
