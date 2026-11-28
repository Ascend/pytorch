# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.symbolic_shapes.ShapeEnv.create_unbacked_symint 接口功能正确性
API 名称：torch.fx.experimental.symbolic_shapes.ShapeEnv.create_unbacked_symint
API 签名：ShapeEnv.create_unbacked_symint(self, source=None) -> SymInt

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | source=None 和 LocalSource |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | None/LocalSource 和无 hint SymInt |
| 传参与不传参 | source 省略及显式 None/LocalSource |
| 等价类/边界值 | 分配唯一性、pending 登记和符号替换不安装 guard |
| 正常传参场景 | 新符号唯一性、无 hint 和 pending 登记 |
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


    def test_create_unbacked_symint_fresh_symbols_have_no_hint_and_are_registered(self):
        env = self._env()
        first = env.create_unbacked_symint()
        second = env.create_unbacked_symint()
        self.assertNotEqual(first.node.expr, second.node.expr)
        self.assertIsNone(first.node.hint)
        self.assertIsNone(second.node.hint)
        self.assertEqual(env.pending_fresh_unbacked_symbols, [first.node.expr, second.node.expr])
        self.assertIn(first.node.expr, env.var_to_range)

    def test_create_unbacked_symint_source_argument_and_symbolic_arithmetic_do_not_add_guards(self):
        env = self._env()
        value = env.create_unbacked_symint(source=LocalSource("unbacked"))
        expression = value * 3 + 2
        self.assertEqual(expression.node.expr.subs(value.node.expr, 4), 14)
        self.assertEqual(expression.node.expr.subs(value.node.expr, 0), 2)
        self.assertEqual(env.guards, [])


    def test_parameter_explicit_none_source_is_registered_without_hint(self):
        env = self._env()
        value = env.create_unbacked_symint(source=None)
        self.assertIsInstance(value, torch.SymInt)
        self.assertIsNone(value.node.hint)
        self.assertIn(value.node.expr, env.pending_fresh_unbacked_symbols)
        self.assertEqual(env.guards, [])


if __name__ == "__main__":
    run_tests()
