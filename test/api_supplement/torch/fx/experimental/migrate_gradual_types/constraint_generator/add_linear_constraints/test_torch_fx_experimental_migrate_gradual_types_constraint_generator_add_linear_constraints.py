# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.migrate_gradual_types.constraint_generator.add_linear_constraints 接口功能正确性
API 名称：torch.fx.experimental.migrate_gradual_types.constraint_generator.add_linear_constraints
API 签名：add_linear_constraints(dims1, dims2, in_features, out_features) ->
    list[torch.fx.experimental.migrate_gradual_types.constraint.Constraint]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：相关场景中的空维度、空分支或空约束 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：维度变量、静态维度、动态类型或真实回调 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：空维度和输入输出 rank 不等 |
| 正常传参场景 | 已覆盖：batch 相等及输入/输出 feature 尺寸约束 |
| 异常传参场景 | 已覆盖：rank 不等抛出 AssertionError |
| 混合设备类型 | 不适用：本批为设备无关的图结构、类型或辅助状态语义 |

未覆盖项及原因：
- 图执行方式：本文件直接验证图模式辅助接口，不声明 NPU Inductor codegen 覆盖。
- eager 对照：使用接口定义的结构、状态或确定值；不构造无意义的 CPU/NPU 自比对。
- autograd：本接口场景不计算梯度。
- RNG：本文件使用确定输入，不依赖随机流。
- 约束场景使用 Z3 的 SAT/UNSAT 作为独立结果断言；不声明数值算子执行或 NPU codegen 覆盖。
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

from torch.fx.experimental.migrate_gradual_types.constraint_generator import add_linear_constraints


import z3
from torch.fx.experimental.migrate_gradual_types.constraint import Conj, DVar
from torch.fx.experimental.migrate_gradual_types.transform_to_z3 import (
    D, iterate_till_fixed_point, transform_dimension, transform_to_z3,
)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def _check(self, constraints, assignments, expected):
        normalized, counter = iterate_till_fixed_point(Conj(constraints), 50)
        dimensions = {}
        formula, counter = transform_to_z3(normalized, counter, dimensions)
        solver = z3.Solver()
        solver.add(formula)
        for variable, concrete in assignments.items():
            encoded, counter = transform_dimension(DVar(variable), counter, dimensions)
            solver.add(encoded == D(1, concrete))
        self.assertEqual(solver.check(), z3.sat if expected else z3.unsat)


    def test_add_linear_constraints_batch_and_feature_dimensions_have_correct_relations(self):
        constraints = add_linear_constraints([DVar(1), DVar(2)], [DVar(3), DVar(4)], 5, 7)
        self._check(constraints, {1: 2, 2: 5, 3: 2, 4: 7}, True)
        self._check(constraints, {1: 2, 2: 4, 3: 2, 4: 7}, False)
        self._check(constraints, {1: 2, 2: 5, 3: 3, 4: 7}, False)
        self._check(constraints, {1: 2, 2: 5, 3: 2, 4: 8}, False)

    def test_add_linear_constraints_empty_dimensions_and_rank_mismatch(self):
        self.assertEqual(add_linear_constraints([], [], 5, 7), [])
        with self.assertRaises(AssertionError):
            add_linear_constraints([DVar(1)], [DVar(2), DVar(3)], 5, 7)


if __name__ == "__main__":
    run_tests()
