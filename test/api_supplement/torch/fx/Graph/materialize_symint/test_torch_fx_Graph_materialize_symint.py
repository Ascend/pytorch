# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.Graph.materialize_symint 接口功能正确性
API 名称：torch.fx.Graph.materialize_symint
API 签名：Graph.materialize_symint(self, value) -> Node | int

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空图、空约束或无依赖状态 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 FX Graph/Node、类型对象及对应接口参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：零/负值、符号计算和缺失 producer |
| 正常传参场景 | 已覆盖：int 保持与有生产节点的符号表达式物化 |
| 异常传参场景 | 已覆盖：float TypeError 及缺失符号 producer 的 AssertionError |
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

from torch.fx.experimental.symbolic_shapes import ShapeEnv


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_materialize_symint_plain_integers_do_not_add_graph_nodes_and_float_is_rejected(self):
        graph = torch.fx.Graph()
        for value in (0, -3, 7):
            with self.subTest(value=value):
                self.assertEqual(graph.materialize_symint(value), value)
        self.assertEqual(list(graph.nodes), [])
        with self.assertRaises(TypeError):
            graph.materialize_symint(2.5)

    def test_materialize_symint_symbolic_expression_is_executable_from_existing_placeholder(self):
        env = ShapeEnv()
        self.addCleanup(env.cleanup)
        value = env.create_unbacked_symint()
        graph = torch.fx.Graph()
        source = graph.placeholder("size")
        source.meta["val"] = value
        self.assertIs(graph.materialize_symint(value), source)
        result = graph.materialize_symint(value * 2 + 1)
        self.assertIsInstance(result, torch.fx.Node)
        graph.output(result)
        graph.lint()
        module = torch.fx.GraphModule({}, graph)
        self.assertEqual(module(3), 7)
        self.assertEqual(module(0), 1)
        with self.assertRaisesRegex(AssertionError, "no producer"):
            graph.materialize_symint(env.create_unbacked_symint())


if __name__ == "__main__":
    run_tests()
