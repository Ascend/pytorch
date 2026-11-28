# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.symbolic_shapes.guard_scalar 接口功能正确性
API 名称：torch.fx.experimental.symbolic_shapes.guard_scalar
API 签名：guard_scalar(a) -> bool | int | float

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 相关场景中的空序列、无绑定或空缓存；其他场景不适用 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | bool/int/float 和 SymBool/SymInt/SymFloat |
| 传参与不传参 | 不适用：a 必传 |
| 等价类/边界值 | 真/假、零/负值、有 hint 符号结果及精确返回类型 |
| 正常传参场景 | bool/int/float 值及精确返回类型 |
| 异常传参场景 | 非法字符串 AssertionError；无 hint 符号值 GuardOnDataDependentSymNode |
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

from torch.fx.experimental.symbolic_shapes import guard_scalar, ShapeEnv
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


    def test_guard_scalar_supported_concrete_types_preserve_values_and_types(self):
        for value in (True, False, 0, -7, 2.5, -0.5):
            with self.subTest(value=value):
                result = guard_scalar(value)
                self.assertEqual(result, value)
                self.assertIs(type(result), type(value))

    def test_guard_scalar_backed_symbolic_integer_and_invalid_scalar(self):
        env = self._env()
        value = env.create_unspecified_symint_and_symbol(7, LocalSource("value"), DimDynamic.DYNAMIC)
        self.assertEqual(guard_scalar(value + 2), 9)
        self.assertTrue(env.guards)
        with self.assertRaises(AssertionError):
            guard_scalar("7")


    def test_parameter_symbolic_boolean_and_float_preserve_concrete_result_types(self):
        from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode

        for expected in (False, True, 0.0, -2.5, 3.25):
            with self.subTest(expected=expected):
                env = self._env()
                if isinstance(expected, bool):
                    integer = env.create_unspecified_symint_and_symbol(5, LocalSource("value"), DimDynamic.DYNAMIC)
                    value = integer > 3 if expected else integer < 3
                else:
                    expression = env.create_unspecified_symbol(expected, LocalSource("value"))
                    value = env.create_symfloatnode(expression, hint=expected)
                result = guard_scalar(value)
                self.assertEqual(result, expected)
                self.assertIs(type(result), type(expected))
                self.assertTrue(env.guards)
        env = self._env()
        for value in (env.create_unbacked_symint() > 3, env.create_unbacked_symfloat()):
            with self.assertRaises(GuardOnDataDependentSymNode):
                guard_scalar(value)


if __name__ == "__main__":
    run_tests()
