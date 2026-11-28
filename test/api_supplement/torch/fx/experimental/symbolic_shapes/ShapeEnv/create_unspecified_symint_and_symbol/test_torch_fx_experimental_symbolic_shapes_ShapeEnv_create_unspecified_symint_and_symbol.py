# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.symbolic_shapes.ShapeEnv.create_unspecified_symint_and_symbol 接口功能正确性
API 名称：torch.fx.experimental.symbolic_shapes.ShapeEnv.create_unspecified_symint_and_symbol
API 签名：ShapeEnv.create_unspecified_symint_and_symbol(self, value, source, dynamic_dim, excluded_value=None) ->
    IntLikeType

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 相关场景的空 guard、空参数或无替换状态 |
| 枚举选项 | DYNAMIC/STATIC/DUCK/UNBACKED，exclusion 配置开/关 |
| 参数类型 | 整数、LocalSource、DimDynamic 和 None/整数 excluded_value |
| 传参与不传参 | excluded_value 省略及显式 None/0 |
| 等价类/边界值 | 0/1/负数、静态 int、无 hint UNBACKED 和排除 guard |
| 正常传参场景 | SymInt 包装和 backing map/hint |
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

from torch._dynamo.source import LocalSource
from torch.fx.experimental.symbolic_shapes import ShapeEnv, DimDynamic
from torch._dynamo import config


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


    def test_create_unspecified_symint_and_symbol_wrapped_symbol_keeps_hint_and_dynamic_zero_one(self):
        env = self._env()
        for index, value in enumerate((0, 1, -3, 7)):
            with self.subTest(value=value):
                result = self._backed(env, value, f"value_{index}")
                self.assertIsInstance(result, torch.SymInt)
                self.assertEqual(result.node.hint, value)
                self.assertEqual(env.backed_var_to_val[result.node.expr], value)

    def test_create_unspecified_symint_and_symbol_excluded_value_changes_generated_runtime_guard(self):
        env = self._env()
        value = env.create_unspecified_symint_and_symbol(
            5, LocalSource("t0"), DimDynamic.DYNAMIC, excluded_value=0)
        with config.patch(automatic_dynamic_exclusion_guard=True, enable_compiler_collectives=False):
            guards = env.produce_guards([value], [LocalSource("t0")])
        self.assertTrue(guards)
        expression = " and ".join(guards)
        self.assertTrue(env.evaluate_guards_expression(expression, [5]))
        self.assertFalse(env.evaluate_guards_expression(expression, [0]))

    def test_create_unspecified_symint_and_symbol_exclusion_feature_disabled_does_not_emit_exclusion_guard(self):
        env = self._env()
        value = env.create_unspecified_symint_and_symbol(
            5, LocalSource("t0"), DimDynamic.DYNAMIC, excluded_value=0)
        with config.patch(automatic_dynamic_exclusion_guard=False):
            self.assertEqual(env.produce_guards([value], [LocalSource("t0")]), [])


    def test_parameter_static_duck_unbacked_and_explicit_none_exclusion(self):
        for policy in (DimDynamic.STATIC, DimDynamic.DUCK, DimDynamic.UNBACKED):
            with self.subTest(policy=policy):
                env = self._env()
                value = env.create_unspecified_symint_and_symbol(7, LocalSource("value"), policy, excluded_value=None)
                if policy is DimDynamic.STATIC:
                    self.assertIs(type(value), int)
                    self.assertEqual(value, 7)
                else:
                    self.assertIsInstance(value, torch.SymInt)
                    if policy is DimDynamic.UNBACKED:
                        self.assertIsNone(value.node.hint)
                        self.assertIn(value.node.expr, env.unbacked_inputs)
                    else:
                        self.assertEqual(value.node.hint, 7)


if __name__ == "__main__":
    run_tests()
