# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.symbolic_shapes.find_symbol_binding_fx_nodes 接口功能正确性
API 名称：torch.fx.experimental.symbolic_shapes.find_symbol_binding_fx_nodes
API 签名：find_symbol_binding_fx_nodes(graph) -> dict[sympy.Symbol, torch.fx.Node]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：相关场景中的空序列、无绑定或空缓存；其他场景不适用 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：场景所需的 Python 标量、真实 SymInt/SymBool、FX 节点或缓存回调 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：空图和无符号绑定的图 |
| 正常传参场景 | 已覆盖：多个符号绑定与重复符号首次节点保留 |
| 异常传参场景 | 不适用：有效符号输入的确定结果 |
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

from torch.fx.experimental.symbolic_shapes import find_symbol_binding_fx_nodes, ShapeEnv


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def _env(self):
        env = ShapeEnv()
        self.addCleanup(env.cleanup)
        return env


    def test_find_symbol_binding_fx_nodes_first_binding_wins_for_repeated_symbol(self):
        env = self._env()
        first = env.create_unbacked_symint()
        second = env.create_unbacked_symint()
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        repeated = graph.placeholder("repeated")
        x.meta["val"], y.meta["val"], repeated.meta["val"] = first, second, first
        self.assertEqual(find_symbol_binding_fx_nodes(graph),
                         {first.node.expr: x, second.node.expr: y})

    def test_find_symbol_binding_fx_nodes_empty_and_non_binding_graphs(self):
        graph = torch.fx.Graph()
        self.assertEqual(find_symbol_binding_fx_nodes(graph), {})
        source = graph.placeholder("source")
        source.meta["val"] = 3
        graph.call_function(abs, (source,))
        self.assertEqual(find_symbol_binding_fx_nodes(graph), {})


if __name__ == "__main__":
    run_tests()
