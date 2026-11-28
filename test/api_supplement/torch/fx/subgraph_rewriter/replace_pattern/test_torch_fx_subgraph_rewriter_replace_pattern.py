# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.subgraph_rewriter.replace_pattern 接口功能正确性
API 名称：torch.fx.subgraph_rewriter.replace_pattern
API 签名：replace_pattern(gm, pattern, replacement) -> list[torch.fx.subgraph_rewriter.Match]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空匹配、空节点或无注解状态 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 GraphModule/Node、模块与默认/显式参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：无匹配空列表和图保持 |
| 正常传参场景 | 已覆盖：多处非重叠匹配替换及改变后的执行语义 |
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

import operator
from torch.fx.subgraph_rewriter import replace_pattern


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_replace_pattern_two_nonoverlapping_matches_are_replaced_and_executable(self):
        module = torch.fx.symbolic_trace(lambda x, y: -x + -y)
        matches = replace_pattern(module, lambda x: -x, lambda x: x * 3)
        self.assertEqual(len(matches), 2)
        self.assertTrue(all(match.nodes_map for match in matches))
        calls = [node.target for node in module.graph.nodes if node.op == "call_function"]
        self.assertNotIn(operator.neg, calls)
        self.assertEqual(calls.count(operator.mul), 2)
        self.assertEqual(module(2, -4), -6)
        self.assertEqual(module(0, 5), 15)
        module.graph.lint()

    def test_replace_pattern_no_match_keeps_execution_and_graph_structure(self):
        module = torch.fx.symbolic_trace(lambda value: value + 1)
        before = [(node.op, node.target) for node in module.graph.nodes]
        matches = replace_pattern(module, lambda x: -x, lambda x: x * 3)
        self.assertEqual(matches, [])
        self.assertEqual([(node.op, node.target) for node in module.graph.nodes], before)
        self.assertEqual(module(-2), -1)


if __name__ == "__main__":
    run_tests()
