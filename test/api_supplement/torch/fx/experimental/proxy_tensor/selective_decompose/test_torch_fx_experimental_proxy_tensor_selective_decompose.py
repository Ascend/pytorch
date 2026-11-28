# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.proxy_tensor.selective_decompose 接口功能正确性
API 名称：torch.fx.experimental.proxy_tensor.selective_decompose
API 签名：selective_decompose(joint_gm, *args, decomposition, should_decompose, trace_joint_graph) -> fx.GraphModule

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | decomposition=None/空/非空映射，选中/未选中节点 |
| 枚举选项 | trace_joint_graph=False/True，节点选择回调的两种结果 |
| 参数类型 | GraphModule、Tensor、分解映射/回调和 primals/tangents 列表 |
| 传参与不传参 | 可变位置实参；三个关键字形参均必传 |
| 等价类/边界值 | 普通/joint 图、选中/未选中和改变输入 |
| 正常传参场景 | 选择性分解、两路 primals/tangents 结果和图结构 |
| 异常传参场景 | 未覆盖：无稳定异常契约 |
| 混合设备类型 | 未覆盖：同一加速器输入，不涉及混合设备语义 |

未覆盖项及原因：
- 图执行方式：直接调用被测 tracing/dispatch 入口并执行返回的 FX 图；未覆盖 Inductor codegen。
- eager 对照：相同设备及输入下验证 ATen 图，包含不同输入的重复执行。
- autograd：未覆盖 backward；本文件验证前向捕获。
- RNG：使用确定输入。
- 参数类型：未覆盖未绑定形状 epoch 分支。
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

from torch.fx.experimental import proxy_tensor as pt


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def _rewrite(self, selected):
        self.assertIsNotNone(self.device)
        value = torch.arange(6, dtype=torch.float32, device=self.device).reshape(2, 3) - 2

        def function(x):
            return torch.ops.aten.neg.default(x) + 2

        original = pt.make_fx(function, decomposition_table={})(value)
        rewritten = pt.selective_decompose(
            original, value, decomposition={torch.ops.aten.neg.default: lambda x: x * -1},
            should_decompose=lambda node: selected and node.target is torch.ops.aten.neg.default,
            trace_joint_graph=False,
        )
        for current in (value, value + 3):
            actual = rewritten(current)
            torch.testing.assert_close(actual.cpu(), function(current).cpu())
            self.assertEqual(actual.device.type, self.device_name)
            self.assertEqual(actual.shape, current.shape)
            self.assertEqual(actual.dtype, current.dtype)
        rewritten.graph.lint()
        return rewritten

    def test_selective_decompose_selected_operator_is_replaced_with_decomposition(self):
        graph = self._rewrite(True)
        targets = [node.target for node in graph.graph.nodes if node.op == "call_function"]
        self.assertNotIn(torch.ops.aten.neg.default, targets)
        self.assertIn(torch.ops.aten.mul.Tensor, targets)

    def test_selective_decompose_unselected_operator_is_preserved(self):
        graph = self._rewrite(False)
        self.assertTrue(any(node.target is torch.ops.aten.neg.default for node in graph.graph.nodes))


    def test_parameter_joint_graph_primals_and_tangents_are_both_decomposed(self):
        value = torch.arange(6, dtype=torch.float32, device=self.device).reshape(2, 3)
        tangent = torch.ones_like(value)

        def joint(primals, tangents):
            return [torch.ops.aten.neg.default(primals[0]), torch.ops.aten.neg.default(tangents[0])]

        original = pt.make_fx(joint, decomposition_table={})([value], [tangent])
        rewritten = pt.selective_decompose(
            original, [value], [tangent], decomposition={torch.ops.aten.neg.default: lambda x: x * -1},
            should_decompose=lambda node: node.target is torch.ops.aten.neg.default, trace_joint_graph=True)
        targets = [node.target for node in rewritten.graph.nodes if node.op == "call_function"]
        self.assertNotIn(torch.ops.aten.neg.default, targets)
        self.assertIn(torch.ops.aten.mul.Tensor, targets)
        for shift in (0, 2):
            actual = rewritten([value + shift], [tangent + shift])
            expected = joint([value + shift], [tangent + shift])
            self.assertEqual(len(actual), 2)
            for observed, golden in zip(actual, expected):
                torch.testing.assert_close(observed.cpu(), golden.cpu())
                self.assertEqual(observed.device.type, self.device_name)
        rewritten.graph.lint()

    def test_parameter_none_and_empty_decomposition_tables_preserve_unselected_graph(self):
        value = torch.arange(6, dtype=torch.float32, device=self.device)

        def function(current):
            return current.sin() + 1

        original = pt.make_fx(function)(value)
        for table in (None, {}):
            seen = []

            def predicate(node):
                seen.append(node.target)
                return False

            rewritten = pt.selective_decompose(original, value, decomposition=table,
                                               should_decompose=predicate, trace_joint_graph=False)
            self.assertTrue(seen)
            for current in (value, value + 1):
                torch.testing.assert_close(rewritten(current).cpu(), function(current).cpu())


if __name__ == "__main__":
    run_tests()
