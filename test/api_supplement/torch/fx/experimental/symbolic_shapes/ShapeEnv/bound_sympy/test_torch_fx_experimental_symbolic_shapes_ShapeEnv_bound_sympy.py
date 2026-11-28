# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.symbolic_shapes.ShapeEnv.bound_sympy 接口功能正确性
API 名称：torch.fx.experimental.symbolic_shapes.ShapeEnv.bound_sympy
API 签名：ShapeEnv.bound_sympy(self, expr, size_oblivious=False) -> ValueRanges[sympy.Expr]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 相关场景的空 guard、空参数或无替换状态 |
| 枚举选项 | size_oblivious=False/True |
| 参数类型 | SymPy 常量、受约束符号、仿射表达式和 size-like 符号 |
| 传参与不传参 | size_oblivious 默认值和显式两值 |
| 等价类/边界值 | 负系数反转、oblivious 区间及原范围保持 |
| 正常传参场景 | 常量与受约束仿射表达式区间 |
| 异常传参场景 | 不适用：有效符号状态操作 |
| 混合设备类型 | 不适用：本文件为设备无关的图结构、类型或辅助状态语义 |

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
from torch.fx.experimental.symbolic_shapes import ShapeEnv, DimDynamic


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def _env(self, **kwargs):
        env = ShapeEnv(**kwargs)
        self.addCleanup(env.cleanup)
        return env

    def _backed(self, env, value=7, name="value"):
        return env.create_unspecified_symint_and_symbol(value, LocalSource(name), DimDynamic.DYNAMIC)


    def test_bound_sympy_constant_and_constrained_affine_expression_bounds(self):
        env = self._env()
        symbol = env.create_unbacked_symint().node.expr
        env.constrain_symbol_range(symbol, 2, 5)
        bound = env.bound_sympy(3 * symbol + 1)
        self.assertEqual((bound.lower, bound.upper), (7, 16))
        constant = env.bound_sympy(sympy.Integer(4))
        self.assertEqual((constant.lower, constant.upper), (4, 4))

    def test_bound_sympy_negative_coefficient_reverses_interval_endpoints(self):
        env = self._env()
        symbol = env.create_unbacked_symint().node.expr
        env.constrain_symbol_range(symbol, -2, 5)
        bound = env.bound_sympy(3 - 2 * symbol, size_oblivious=False)
        self.assertEqual((bound.lower, bound.upper), (-7, 7))
        self.assertEqual(env.guards, [])


    def test_parameter_size_oblivious_true_changes_size_like_bounds(self):
        from torch.utils._sympy.numbers import int_oo

        env = self._env()
        symbol = env.create_unbacked_symint().node.expr
        env._constrain_range_for_size(symbol, min=0, max=5)
        ordinary = env.bound_sympy(symbol, size_oblivious=False)
        oblivious = env.bound_sympy(symbol, size_oblivious=True)
        self.assertEqual((ordinary.lower, ordinary.upper), (0, 5))
        self.assertEqual(oblivious.lower, 2)
        self.assertTrue(oblivious.upper == int_oo)
        self.assertEqual((env.var_to_range[symbol].lower, env.var_to_range[symbol].upper), (0, 5))


if __name__ == "__main__":
    run_tests()
