# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_sym_node 接口功能正确性
API 名称：torch.fx.experimental.symbolic_shapes.ShapeEnv.evaluate_sym_node
API 签名：ShapeEnv.evaluate_sym_node(self, sym_node, size_oblivious=False, fallback_value=None) -> sympy.Basic

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | fallback_value=None 和布尔值 |
| 枚举选项 | size_oblivious=False/True；fallback_value=None/False/True |
| 参数类型 | backed/unbacked SymNode、符号布尔节点和布尔 fallback |
| 传参与不传参 | 两个可选参数省略及显式值 |
| 等价类/边界值 | 符号常量、无 hint、size-like 谓词和精确布尔 fallback |
| 正常传参场景 | backed SymNode 算术求值与 guard 安装 |
| 异常传参场景 | 数据相关节点求值抛出专用错误 |
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
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode


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


    def test_evaluate_sym_node_backed_symbolic_arithmetic_evaluation_installs_guard(self):
        env = self._env()
        value = self._backed(env, 7)
        self.assertEqual(env.evaluate_sym_node((value + 2).node), 9)
        self.assertTrue(env.guards)

    def test_evaluate_sym_node_constant_boolean_and_data_dependent_node(self):
        env = self._env()
        self.assertEqual(env.evaluate_sym_node(env.create_symboolnode(sympy.true).node), sympy.true)
        value = env.create_unbacked_symint()
        with self.assertRaises(GuardOnDataDependentSymNode):
            env.evaluate_sym_node(value.node)


    def test_parameter_both_fallback_values_and_explicit_none(self):
        env = self._env()
        value = env.create_unbacked_symint()
        condition = (value > 3).node
        for fallback in (False, True):
            with self.subTest(fallback=fallback):
                self.assertIs(env.evaluate_sym_node(condition, size_oblivious=False,
                                                    fallback_value=fallback), fallback)
        with self.assertRaises(GuardOnDataDependentSymNode):
            env.evaluate_sym_node(condition, fallback_value=None)

    def test_parameter_size_oblivious_evaluates_known_size_lower_bound(self):
        env = self._env()
        value = env.create_unbacked_symint()
        env._constrain_range_for_size(value.node.expr, min=0, max=5)
        self.assertEqual(env.evaluate_sym_node((value >= 2).node, size_oblivious=True), sympy.true)


if __name__ == "__main__":
    run_tests()
