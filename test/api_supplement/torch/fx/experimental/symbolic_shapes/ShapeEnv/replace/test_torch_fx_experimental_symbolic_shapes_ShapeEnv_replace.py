# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.symbolic_shapes.ShapeEnv.replace 接口功能正确性
API 名称：torch.fx.experimental.symbolic_shapes.ShapeEnv.replace
API 签名：ShapeEnv.replace(self, expr) -> _SympyT

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：相关场景的空 guard、空参数或无替换状态 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 ShapeEnv、Source、符号表达式和具体 hint；详见本文件场景 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：传递替换与无关符号保持 |
| 正常传参场景 | 已覆盖：实际 replacement 对已缓存表达式的更新 |
| 异常传参场景 | 不适用：有效符号状态操作 |
| 混合设备类型 | 不适用：本批为设备无关的图结构、类型或辅助状态语义 |

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


    def test_replace_replacement_changes_cached_symbolic_expression(self):
        env = self._env()
        symbol = env.create_unbacked_symint().node.expr
        expression = 3 * symbol + 2
        self.assertEqual(env.replace(expression), expression)
        env._set_replacement(symbol, sympy.Integer(4), "unit test equality")
        self.assertEqual(env.replace(expression), 14)

    def test_replace_transitive_replacement_and_unaffected_symbol(self):
        env = self._env()
        first = env.create_unbacked_symint().node.expr
        second = env.create_unbacked_symint().node.expr
        other = env.create_unbacked_symint().node.expr
        env._set_replacement(first, second + 1, "unit test relation")
        env._set_replacement(second, sympy.Integer(4), "unit test value")
        self.assertEqual(env.replace(2 * first), 10)
        self.assertEqual(env.replace(other + 2), other + 2)


if __name__ == "__main__":
    run_tests()
