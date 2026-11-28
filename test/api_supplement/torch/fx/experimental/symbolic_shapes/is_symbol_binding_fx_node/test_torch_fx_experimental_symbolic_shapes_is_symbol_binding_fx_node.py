# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.symbolic_shapes.is_symbol_binding_fx_node 接口功能正确性
API 名称：torch.fx.experimental.symbolic_shapes.is_symbol_binding_fx_node
API 签名：is_symbol_binding_fx_node(node) -> sympy.Symbol | None

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：相关场景中的空序列、无绑定或空缓存；其他场景不适用 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：场景所需的 Python 标量、真实 SymInt/SymBool、FX 节点或缓存回调 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：缺失值、常量、复合表达式与无绑定节点 |
| 正常传参场景 | 已覆盖：符号 placeholder 和显式 unbacked 绑定节点 |
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

from torch.fx.experimental.symbolic_shapes import is_symbol_binding_fx_node, ShapeEnv


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


    def test_is_symbol_binding_fx_node_symbol_placeholder_and_explicit_unbacked_binding(self):
        env = self._env()
        symbol = env.create_unbacked_symint()
        graph = torch.fx.Graph()
        source = graph.placeholder("symbol")
        source.meta["val"] = symbol
        self.assertEqual(is_symbol_binding_fx_node(source), symbol.node.expr)
        result = graph.call_function(abs, (source,))
        result.meta["val"] = symbol
        result.meta["unbacked_bindings"] = {symbol.node.expr: ()}
        self.assertEqual(is_symbol_binding_fx_node(result), symbol.node.expr)

    def test_is_symbol_binding_fx_node_missing_non_symbol_and_non_binding_nodes(self):
        env = self._env()
        graph = torch.fx.Graph()
        source = graph.placeholder("value")
        self.assertIsNone(is_symbol_binding_fx_node(source))
        source.meta["val"] = 4
        self.assertIsNone(is_symbol_binding_fx_node(source))
        source.meta["val"] = env.create_unbacked_symint() + 1
        self.assertIsNone(is_symbol_binding_fx_node(source))
        result = graph.call_function(abs, (source,))
        result.meta["val"] = env.create_unbacked_symint()
        self.assertIsNone(is_symbol_binding_fx_node(result))


if __name__ == "__main__":
    run_tests()
