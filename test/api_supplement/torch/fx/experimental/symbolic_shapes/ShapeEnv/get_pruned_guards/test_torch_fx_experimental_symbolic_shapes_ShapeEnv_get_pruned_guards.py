# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.symbolic_shapes.ShapeEnv.get_pruned_guards 接口功能正确性
API 名称：torch.fx.experimental.symbolic_shapes.ShapeEnv.get_pruned_guards
API 签名：ShapeEnv.get_pruned_guards(self, symints) -> list[ShapeGuard]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：相关场景的空 guard、空参数或无替换状态 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 ShapeEnv、Source、符号表达式和具体 hint；详见本文件场景 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：部分符号、无关符号和空输入 |
| 正常传参场景 | 已覆盖：输入符号覆盖全部自由符号时才保留 guard |
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


    def test_get_pruned_guards_only_guards_whose_symbols_are_all_in_inputs_are_kept(self):
        env = self._env()
        left = self._backed(env, 9, "left")
        right = self._backed(env, 5, "right")
        self.assertTrue(bool(left > right))
        self.assertEqual(env.get_pruned_guards([left]), [])
        selected = env.get_pruned_guards([left, right])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].expr.subs({left.node.expr: 9, right.node.expr: 5}), sympy.true)

    def test_get_pruned_guards_empty_and_unrelated_symbol_inputs(self):
        env = self._env()
        value = self._backed(env, 7)
        other = self._backed(env, 9, "other")
        self.assertTrue(bool(value > 3))
        before = list(env.guards)
        self.assertEqual(env.get_pruned_guards([]), [])
        self.assertEqual(env.get_pruned_guards([other]), [])
        self.assertEqual(env.get_pruned_guards([value]), before)
        self.assertEqual(env.guards, before)


if __name__ == "__main__":
    run_tests()
