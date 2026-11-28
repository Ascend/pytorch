# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symfloatnode 接口功能正确性
API 名称：torch.fx.experimental.symbolic_shapes.ShapeEnv.create_symfloatnode
API 签名：ShapeEnv.create_symfloatnode(self, sym, *, hint, source=None) -> FloatLikeType

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | hint/source 的 None 和实际值 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | SymPy 浮点表达式，None/数字/SymInt hint，LocalSource |
| 传参与不传参 | source 省略、None/LocalSource；必传 hint 的 None/显式数值 |
| 等价类/边界值 | 常量、冲突 hint 和翻译验证的 source placeholder |
| 正常传参场景 | 常量 float 与实际登记的符号浮点表达式 |
| 异常传参场景 | 常量与 hint 不一致抛出 AssertionError |
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
from torch.fx.experimental.symbolic_shapes import guard_float


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


    def test_create_symfloatnode_float_constants_and_inconsistent_hint(self):
        env = self._env()
        self.assertEqual(env.create_symfloatnode(sympy.Float(2.5), hint=None), 2.5)
        self.assertEqual(env.create_symfloatnode(sympy.Float(-1.5), hint=-1.5), -1.5)
        with self.assertRaises(AssertionError):
            env.create_symfloatnode(sympy.Float(2.5), hint=3.5)

    def test_create_symfloatnode_tracked_float_expression_is_symbolic_and_evaluates_with_hint(self):
        env = self._env()
        symbol = env.create_unspecified_symbol(2.5, LocalSource("scalar"))
        value = env.create_symfloatnode(2 * symbol, hint=5.0)
        self.assertIsInstance(value, torch.SymFloat)
        self.assertEqual(value.node.expr, 2 * symbol)
        self.assertEqual(guard_float(value), 5.0)


    def test_parameter_hint_types_and_explicit_source_none(self):
        env = self._env()
        for hint in (None, 1, 1.0, True):
            with self.subTest(hint=hint):
                value = env.create_symfloatnode(sympy.Float(1.0), hint=hint, source=None)
                self.assertEqual(value, 1.0)
                self.assertIsInstance(value, float)
        integer_hint = self._backed(env, 3)
        self.assertEqual(env.create_symfloatnode(sympy.Float(3.0), hint=integer_hint, source=None), 3.0)
        with self.assertRaises(AssertionError):
            env.create_symfloatnode(sympy.Float(1.0), hint=False, source=None)

    def test_parameter_non_none_source_records_translation_validation_placeholder(self):
        from torch.fx.experimental import _config

        with _config.patch(translation_validation=True):
            env = self._env()
            symbol = env.create_unspecified_symbol(2.5, LocalSource("original_float"))
            without_source = env.create_symfloatnode(2 * symbol, hint=5.0, source=None)
            with_source = env.create_symfloatnode(2 * symbol, hint=5.0, source=LocalSource("result_float"))
            self.assertIsNone(without_source.node.fx_node)
            self.assertIsNotNone(with_source.node.fx_node)
            self.assertEqual(with_source.node.fx_node.op, "placeholder")
            self.assertEqual(guard_float(with_source), 5.0)
            self.assertTrue(with_source.node.expr == 2 * symbol)


if __name__ == "__main__":
    run_tests()
