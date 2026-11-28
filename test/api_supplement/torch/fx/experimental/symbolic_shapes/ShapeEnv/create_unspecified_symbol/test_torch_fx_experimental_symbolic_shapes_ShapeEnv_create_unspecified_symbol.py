# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.symbolic_shapes.ShapeEnv.create_unspecified_symbol 接口功能正确性
API 名称：torch.fx.experimental.symbolic_shapes.ShapeEnv.create_unspecified_symbol
API 签名：ShapeEnv.create_unspecified_symbol(self, val, source, dynamic_dim=DimDynamic.DUCK, constraint_dim=None,
    symbolic_context=None) -> sympy.Expr

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 相关场景的空 guard、空参数或无替换状态 |
| 枚举选项 | DYNAMIC/DUCK/STATIC/UNBACKED，None/INFER_STRIDE 非法策略 |
| 参数类型 | 整数、LocalSource、DimDynamic 和 None/StatefulSymbolicContext |
| 传参与不传参 | 动态策略默认/显式值，constraint_dim 默认/None，context 默认/None/状态上下文 |
| 等价类/边界值 | 0/1/负数、符号共享/独立、STATIC/UNBACKED 和来源缓存 |
| 正常传参场景 | 0/1/负数不专化及真实 hint 登记 |
| 异常传参场景 | None/INFER_STRIDE 动态尺寸策略 AssertionError |
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


    def test_create_unspecified_symbol_zero_one_negative_values_are_not_specialized(self):
        env = self._env()
        for index, value in enumerate((0, 1, -3)):
            with self.subTest(value=value):
                symbol = env.create_unspecified_symbol(value, LocalSource(f"value_{index}"), DimDynamic.DYNAMIC)
                self.assertIsInstance(symbol, sympy.Symbol)
                self.assertEqual(env.guarding_hint_or_throw(symbol), value)
        self.assertEqual(env.guards, [])

    def test_create_unspecified_symbol_dynamic_and_duck_symbol_sharing_policies(self):
        env = self._env(duck_shape=True)
        first = env.create_unspecified_symbol(4, LocalSource("first"), DimDynamic.DYNAMIC)
        second = env.create_unspecified_symbol(4, LocalSource("second"), DimDynamic.DYNAMIC)
        self.assertNotEqual(first, second)
        duck_first = env.create_unspecified_symbol(6, LocalSource("duck_first"), DimDynamic.DUCK)
        duck_second = env.create_unspecified_symbol(6, LocalSource("duck_second"), DimDynamic.DUCK)
        self.assertEqual(duck_first, duck_second)


    def test_parameter_stateful_context_reuses_source_specific_symbol(self):
        from torch.fx.experimental.symbolic_shapes import StatefulSymbolicContext

        env = self._env()
        context = StatefulSymbolicContext([DimDynamic.DYNAMIC], tensor_source=LocalSource("tensor"))
        first = env.create_unspecified_symbol(7, LocalSource("value"), DimDynamic.DYNAMIC,
                                              constraint_dim=None, symbolic_context=context)
        second = env.create_unspecified_symbol(7, LocalSource("value"), DimDynamic.DYNAMIC,
                                               symbolic_context=context)
        self.assertIs(first, second)
        self.assertEqual(context.shape_env_to_source_to_symbol_cache[id(env)]["L['value']"], first)
        independent = env.create_unspecified_symbol(7, LocalSource("fresh"), DimDynamic.DYNAMIC,
                                                    symbolic_context=None)
        self.assertNotEqual(independent, first)
        self.assertEqual(env.guarding_hint_or_throw(first), 7)


    def test_parameter_static_unbacked_and_invalid_stride_only_policies(self):
        env = self._env()
        for number in (0, 1, 7):
            with self.subTest(number=number):
                static = env.create_unspecified_symbol(number, LocalSource("static" + str(number)),
                                                       DimDynamic.STATIC, constraint_dim=None, symbolic_context=None)
                self.assertEqual(static, sympy.Integer(number))
                unbacked = env.create_unspecified_symbol(number, LocalSource("unbacked" + str(number)),
                                                         DimDynamic.UNBACKED)
                self.assertIsInstance(unbacked, sympy.Symbol)
                self.assertIn(unbacked, env.unbacked_inputs)
                self.assertNotIn(unbacked, env.backed_var_to_val)
        for policy in (None, DimDynamic.INFER_STRIDE):
            with self.assertRaisesRegex(AssertionError, "unhandled dynamic_dim"):
                env.create_unspecified_symbol(4, LocalSource("invalid"), policy)


if __name__ == "__main__":
    run_tests()
