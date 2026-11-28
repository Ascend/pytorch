# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.symbolic_shapes.DimConstraints.forced_specializations 接口功能正确性
API 名称：torch.fx.experimental.symbolic_shapes.DimConstraints.forced_specializations
API 签名：DimConstraints.forced_specializations(self) -> dict[str, sympy.Expr]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空状态、约束或记录 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 ShapeEnv、SymPy 表达式及对应默认/显式参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：空结果、静态符号排除及无调试名称 |
| 正常传参场景 | 已覆盖：求解后动态符号特化及调试名称 |
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

import sympy
from torch._dynamo.source import LocalSource
from torch.fx.experimental.symbolic_shapes import DimConstraints


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def _constraints(self, debug=True, dynamic=True):
        symbol = sympy.Symbol("s_ut", integer=True, positive=True)
        source = LocalSource("batch")
        constraints = DimConstraints({symbol: [source]}, {symbol: sympy.Integer(6)},
                                     {symbol} if dynamic else set(), {source.name: "batch"} if debug else {})
        return constraints, symbol, source

    def test_forced_specializations_solved_dynamic_specialization_includes_debug_and_source_name(self):
        constraints, symbol, source = self._constraints()
        self.assertEqual(constraints.forced_specializations(), {})
        constraints.add(sympy.Eq(symbol, 6))
        constraints.solve()
        self.assertEqual(constraints.forced_specializations(), {"batch = " + source.name: sympy.Integer(6)})

    def test_forced_specializations_unmarked_static_symbols_are_excluded_and_no_debug_name_is_supported(self):
        constraints, symbol, _ = self._constraints(dynamic=False)
        constraints.add(sympy.Eq(symbol, 6))
        constraints.solve()
        self.assertEqual(constraints.forced_specializations(), {})
        other, symbol, source = self._constraints(debug=False)
        other.add(sympy.Eq(symbol, 6))
        other.solve()
        self.assertEqual(other.forced_specializations(), {source.name: sympy.Integer(6)})


if __name__ == "__main__":
    run_tests()
