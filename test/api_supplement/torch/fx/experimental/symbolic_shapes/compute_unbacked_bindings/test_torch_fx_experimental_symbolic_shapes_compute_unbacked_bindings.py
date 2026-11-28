# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings 接口功能正确性
API 名称：torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings
API 签名：compute_unbacked_bindings(shape_env, example_value, old_example_value=None, peek=False) -> dict[sympy.Symbol,
    pytree.KeyPath] | None

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | None/空/非空返回值，无 ShapeEnv 和 old_example_value=None |
| 枚举选项 | peek=False/True |
| 参数类型 | ShapeEnv、符号/普通标量、嵌套容器和符号/整数旧值 |
| 传参与不传参 | old_example_value/peek 省略及显式 None/符号/整数/布尔选项 |
| 等价类/边界值 | pending 消耗/保留、再次调用、符号重绑定和整数替换 |
| 正常传参场景 | 嵌套返回结构的真实 KeyPath 和 pending 消耗 |
| 异常传参场景 | pending 符号缺少绑定时 PendingUnbackedSymbolNotFound |
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

from torch.fx.experimental.symbolic_shapes import compute_unbacked_bindings, ShapeEnv
from torch.utils._pytree import key_get


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


    def test_compute_unbacked_bindings_nested_key_paths_and_consumed_pending_symbols(self):
        env = self._env()
        first = env.create_unbacked_symint()
        second = env.create_unbacked_symint()
        values = (first, [second])
        bindings = compute_unbacked_bindings(env, values)
        self.assertEqual(set(bindings), {first.node.expr, second.node.expr})
        for symbol, path in bindings.items():
            self.assertEqual(key_get(values, path).node.expr, symbol)
        self.assertEqual(env.pending_fresh_unbacked_symbols, [])
        self.assertIsNone(compute_unbacked_bindings(env, values))

    def test_compute_unbacked_bindings_peek_does_not_consume_and_no_environment(self):
        env = self._env()
        value = env.create_unbacked_symint()
        before = list(env.pending_fresh_unbacked_symbols)
        peeked = compute_unbacked_bindings(env, (value,), peek=True)
        self.assertEqual(env.pending_fresh_unbacked_symbols, before)
        self.assertEqual(compute_unbacked_bindings(env, (value,)), peeked)
        self.assertIsNone(compute_unbacked_bindings(None, (3,)))


    def test_parameter_old_example_rebinds_symbolic_and_concrete_values_with_both_peek_options(self):
        import sympy

        for old_kind in ("symbolic", 0, -3, 7):
            for peek in (False, True):
                with self.subTest(old_kind=old_kind, peek=peek):
                    env = self._env()
                    if old_kind == "symbolic":
                        old = env.create_unbacked_symint()
                        compute_unbacked_bindings(env, (old,))
                        expected = old.node.expr
                    else:
                        old = old_kind
                        expected = sympy.Integer(old)
                    fresh = env.create_unbacked_symint()
                    fresh_symbol = fresh.node.expr
                    pending = list(env.pending_fresh_unbacked_symbols)
                    bindings = compute_unbacked_bindings(env, (fresh,), old_example_value=(old,), peek=peek)
                    self.assertEqual(set(bindings), {fresh_symbol})
                    self.assertIs(key_get((fresh,), bindings[fresh_symbol]), fresh)
                    self.assertTrue(env.replace(fresh_symbol) == env.replace(expected))
                    self.assertEqual(env.pending_fresh_unbacked_symbols, pending if peek else [])

    def test_parameter_empty_plain_values_explicit_none_and_missing_pending_binding(self):
        from torch.fx.experimental import _config
        from torch.fx.experimental.symbolic_shapes import PendingUnbackedSymbolNotFound

        for value in (None, 0, 2.5, True, [], (), {}, (None, [])):
            for peek in (False, True):
                with self.subTest(value=value, peek=peek):
                    env = self._env()
                    self.assertIsNone(compute_unbacked_bindings(env, value, old_example_value=None, peek=peek))
        env = self._env()
        env.create_unbacked_symint()
        with _config.patch(soft_pending_unbacked_not_found_error=False):
            with self.assertRaises(PendingUnbackedSymbolNotFound):
                compute_unbacked_bindings(env, None, old_example_value=None, peek=False)


if __name__ == "__main__":
    run_tests()
