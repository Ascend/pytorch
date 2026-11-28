# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.symbolic_shapes.ShapeEnv.patch_source_specialization 接口功能正确性
API 名称：torch.fx.experimental.symbolic_shapes.ShapeEnv.patch_source_specialization
API 签名：ShapeEnv.patch_source_specialization(self, source, check_fn) -> Generator[None, None, None]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：相关场景的空 guard、空参数或无替换状态 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 ShapeEnv、Source、符号表达式和具体 hint；详见本文件场景 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：异常退出的 axioms 和 replacement 恢复 |
| 正常传参场景 | 已覆盖：临时相等专化、冻结及 replacement 恢复 |
| 异常传参场景 | 已覆盖：上下文体异常传播并恢复状态 |
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


    def test_patch_source_specialization_temporary_equality_specialization_restores_replacements(self):
        env = self._env()
        source = LocalSource("value")
        value = self._backed(env, 7)
        symbol = value.node.expr
        before = dict(env.replacements)
        with env.patch_source_specialization(source, lambda symbol: symbol == 7):
            self.assertTrue(env.frozen)
            self.assertEqual(env.replacements[symbol], 7)
        self.assertFalse(env.frozen)
        self.assertEqual(env.replacements, before)

    def test_patch_source_specialization_specialization_state_is_removed_on_exception(self):
        env = self._env()
        source = LocalSource("value")
        self._backed(env, 7)
        axioms = dict(env.axioms)
        replacements = dict(env.replacements)
        with self.assertRaisesRegex(ValueError, "specialized body"):
            with env.patch_source_specialization(source, lambda symbol: symbol == 7):
                raise ValueError("specialized body")
        self.assertFalse(env.frozen)
        self.assertEqual(env.axioms, axioms)
        self.assertEqual(env.replacements, replacements)


if __name__ == "__main__":
    run_tests()
