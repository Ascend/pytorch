# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.symbolic_shapes.has_guarding_hint 接口功能正确性
API 名称：torch.fx.experimental.symbolic_shapes.has_guarding_hint
API 签名：has_guarding_hint(a) -> bool

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 相关场景中的空序列、无绑定或空缓存；其他场景不适用 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | Python int/bool/float/complex 和 SymInt/SymBool/SymFloat |
| 传参与不传参 | 不适用：a 必传 |
| 等价类/边界值 | backed/unbacked 符号、派生谓词和无新增 guard |
| 正常传参场景 | 具体标量和有 hint 的符号值 |
| 异常传参场景 | 不适用：有效符号输入的确定结果 |
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

from torch.fx.experimental.symbolic_shapes import has_guarding_hint, ShapeEnv
from torch._dynamo.source import LocalSource
from torch.fx.experimental.symbolic_shapes import DimDynamic


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


    def test_has_guarding_hint_concrete_and_backed_symbols_have_hint(self):
        self.assertTrue(has_guarding_hint(0))
        self.assertTrue(has_guarding_hint(False))
        env = self._env()
        value = env.create_unspecified_symint_and_symbol(5, LocalSource("value"), DimDynamic.DYNAMIC)
        self.assertTrue(has_guarding_hint(value))
        self.assertTrue(has_guarding_hint(value > 3))

    def test_has_guarding_hint_unbacked_symbol_and_derived_predicate_have_no_hint(self):
        env = self._env()
        value = env.create_unbacked_symint()
        before = list(env.guards)
        self.assertFalse(has_guarding_hint(value))
        self.assertFalse(has_guarding_hint(value > 3))
        self.assertEqual(env.guards, before)


    def test_parameter_python_and_symbolic_float_hints_are_non_guarding(self):
        for value in (0.0, -2.5, 3 + 2j):
            self.assertTrue(has_guarding_hint(value))
        env = self._env()
        expression = env.create_unspecified_symbol(2.5, LocalSource("float_value"))
        backed = env.create_symfloatnode(expression, hint=2.5)
        unbacked = env.create_unbacked_symfloat()
        self.assertTrue(has_guarding_hint(backed))
        self.assertFalse(has_guarding_hint(unbacked))
        self.assertEqual(env.guards, [])


if __name__ == "__main__":
    run_tests()
