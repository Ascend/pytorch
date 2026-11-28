# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.symbolic_shapes.guarding_hint_or_throw 接口功能正确性
API 名称：torch.fx.experimental.symbolic_shapes.guarding_hint_or_throw
API 签名：guarding_hint_or_throw(a) -> int | bool

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 相关场景中的空序列、无绑定或空缓存；其他场景不适用 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | int/bool、SymInt/SymBool 和直接 SymNode |
| 传参与不传参 | 不适用：a 必传 |
| 等价类/边界值 | hint 缓存、真/假、无 hint 和无新增 guard |
| 正常传参场景 | 具体整数/布尔值及带 hint 的符号表达式 |
| 异常传参场景 | 数据相关无 hint 错误及不支持类型的 AssertionError |
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

from torch.fx.experimental.symbolic_shapes import guarding_hint_or_throw, ShapeEnv
from torch._dynamo.source import LocalSource
from torch.fx.experimental.symbolic_shapes import DimDynamic, GuardOnDataDependentSymNode


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


    def test_guarding_hint_or_throw_python_and_backed_symbolic_hints(self):
        self.assertEqual(guarding_hint_or_throw(-2), -2)
        self.assertIs(guarding_hint_or_throw(False), False)
        env = self._env()
        value = env.create_unspecified_symint_and_symbol(6, LocalSource("value"), DimDynamic.DYNAMIC)
        self.assertEqual(guarding_hint_or_throw(value + 2), 8)
        self.assertTrue(guarding_hint_or_throw(value > 3))

    def test_guarding_hint_or_throw_unhinted_symbol_and_invalid_type_raise(self):
        env = self._env()
        with self.assertRaises(GuardOnDataDependentSymNode):
            guarding_hint_or_throw(env.create_unbacked_symint())
        with self.assertRaises(AssertionError):
            guarding_hint_or_throw(2.5)


    def test_parameter_direct_symnode_input_and_hint_caching_without_guards(self):
        env = self._env()
        integer = env.create_unspecified_symint_and_symbol(6, LocalSource("value"), DimDynamic.DYNAMIC)
        for value, expected in ((integer + 2, 8), (integer > 3, True), (integer < 3, False)):
            with self.subTest(expected=expected):
                result = guarding_hint_or_throw(value.node)
                self.assertEqual(result, expected)
                self.assertIs(type(result), type(expected))
                self.assertEqual(guarding_hint_or_throw(value.node), expected)
        self.assertEqual(env.guards, [])
        with self.assertRaises(GuardOnDataDependentSymNode):
            guarding_hint_or_throw(env.create_unbacked_symint().node)


if __name__ == "__main__":
    run_tests()
