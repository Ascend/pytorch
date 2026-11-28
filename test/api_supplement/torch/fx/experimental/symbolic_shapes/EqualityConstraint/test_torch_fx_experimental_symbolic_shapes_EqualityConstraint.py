# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.symbolic_shapes.EqualityConstraint 接口功能正确性
API 名称：torch.fx.experimental.symbolic_shapes.EqualityConstraint
API 签名：EqualityConstraint(warn_only, source_pairs, derived_equalities, phantom_symbols, relaxed_sources)

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空状态、约束或记录 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 ShapeEnv、SymPy 表达式及对应默认/显式参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：phantom 根、宽松源与空约束 |
| 正常传参场景 | 已覆盖：并查集传递相等与派生表达式 |
| 异常传参场景 | 未覆盖：无稳定异常契约 |
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
from torch.fx.experimental.symbolic_shapes import EqualityConstraint


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_equalityconstraint_transitive_sources_and_derived_equalities(self):
        a, b, c, d, e, independent = [LocalSource(name) for name in ("a", "b", "c", "d", "e", "other")]
        constraint = EqualityConstraint(
            warn_only=False, source_pairs=[(a, b), (b, c)],
            derived_equalities=[(d, c, lambda x: x + 1), (e, d, lambda x: x - 1)],
            phantom_symbols=[], relaxed_sources=set(),
        )
        self.assertTrue(constraint.is_equal(a, c))
        self.assertTrue(constraint.is_equal(e, a))
        self.assertTrue(constraint.is_derived(d, a, lambda x: x + 1))
        self.assertFalse(constraint.is_equal(d, a))
        self.assertFalse(constraint.is_equal(a, independent))

    def test_equalityconstraint_phantom_roots_relaxed_sources_and_empty_constraints(self):
        a, b = LocalSource("a"), LocalSource("b")
        phantom = sympy.Symbol("phantom", integer=True)
        constraint = EqualityConstraint(
            warn_only=True, source_pairs=[], derived_equalities=[(a, phantom, lambda x: 2 * x + 1)],
            phantom_symbols=[phantom], relaxed_sources={b},
        )
        self.assertTrue(constraint.is_equal(a, b))
        self.assertEqual(constraint._rewrite(a), 2 * phantom + 1)
        empty = EqualityConstraint(
            warn_only=False, source_pairs=[], derived_equalities=[], phantom_symbols=[], relaxed_sources=set(),
        )
        self.assertTrue(empty.is_equal(a, a))
        self.assertFalse(empty.is_equal(a, b))


if __name__ == "__main__":
    run_tests()
