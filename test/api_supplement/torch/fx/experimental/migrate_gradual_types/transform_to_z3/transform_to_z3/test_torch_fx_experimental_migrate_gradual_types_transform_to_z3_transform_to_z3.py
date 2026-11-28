# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.migrate_gradual_types.transform_to_z3.transform_to_z3 接口功能正确性
API 名称：torch.fx.experimental.migrate_gradual_types.transform_to_z3.transform_to_z3
API 签名：transform_to_z3(constraint, counter, dimension_dict) -> tuple[typing.Any, int]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空逻辑约束或映射；其他场景不适用 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：真实渐进类型约束、Z3 表达式、静态与动态维度 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：真/假常量、空合取/析取及混合分支 |
| 正常传参场景 | 已覆盖：维度相等和嵌套算术约束的 SAT/UNSAT |
| 异常传参场景 | 不适用：有效类型或符号表达式 |
| 混合设备类型 | 不适用：本批为设备无关的图结构、类型或辅助状态语义 |

未覆盖项及原因：
- 图执行方式：本文件直接验证图模式辅助接口，不声明 NPU Inductor codegen 覆盖。
- eager 对照：使用接口定义的结构、状态或确定值；不构造无意义的 CPU/NPU 自比对。
- autograd：本接口场景不计算梯度。
- RNG：本文件使用确定输入，不依赖随机流。
- 以 SMT 可满足性或可证明的形状作为 oracle，不声明数值算子或 NPU codegen 覆盖。
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

import z3
from torch.fx.experimental.migrate_gradual_types.transform_to_z3 import transform_to_z3

from torch.fx.experimental.migrate_gradual_types.constraint import BinConstraintD, Conj, Disj, DVar, T, F
from torch.fx.experimental.migrate_gradual_types.operation import op_add, op_eq


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def _sat(self, formula, expected):
        solver = z3.Solver()
        solver.add(formula)
        self.assertEqual(solver.check(), z3.sat if expected else z3.unsat)


    def test_transform_to_z3_dimension_equality_and_arithmetic_constraints(self):
        variable = DVar(3)
        constraint = Conj([BinConstraintD(variable, 3, op_eq),
                           BinConstraintD(BinConstraintD(variable, 2, op_add), 5, op_eq)])
        formula, counter = transform_to_z3(constraint, 20, {})
        self._sat(formula, True)
        self._sat(z3.And(formula, z3.Int(3) == 4), False)
        self.assertGreaterEqual(counter, 21)

    def test_transform_to_z3_boolean_constants_empty_conjunction_and_disjunction(self):
        cases = [(T(), True), (F(), False), (Conj([]), True), (Disj([]), False), (Disj([F(), T()]), True)]
        for constraint, expected in cases:
            with self.subTest(constraint=constraint):
                formula, counter = transform_to_z3(constraint, 20, {})
                self._sat(formula, expected)
                self.assertEqual(counter, 20)


if __name__ == "__main__":
    run_tests()
