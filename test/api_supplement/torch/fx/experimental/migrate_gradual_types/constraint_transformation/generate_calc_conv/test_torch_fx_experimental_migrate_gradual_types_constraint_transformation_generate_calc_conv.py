# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.migrate_gradual_types.constraint_transformation.generate_calc_conv 接口功能正确性
API 名称：torch.fx.experimental.migrate_gradual_types.constraint_transformation.generate_calc_conv
API 签名：generate_calc_conv(constraint, counter) ->
    tuple[torch.fx.experimental.migrate_gradual_types.constraint.Constraint, int]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 未覆盖：本文件以非空形状或有效逻辑约束为主 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：对应的实际 Constraint、TensorType、静态/动态维度与计数器 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：int/tuple 参数及不同轴的配置 |
| 正常传参场景 | 已覆盖：N/C 与空间尺寸的确定结果 |
| 异常传参场景 | 未覆盖：错误约束类的异常；本文件检查正确类型下的 SAT/UNSAT 或唯一结果 |
| 混合设备类型 | 不适用：本批为设备无关的图结构、类型或辅助状态语义 |

未覆盖项及原因：
- 图执行方式：本文件直接验证图模式辅助接口，不声明 NPU Inductor codegen 覆盖。
- eager 对照：使用接口定义的结构、状态或确定值；不构造无意义的 CPU/NPU 自比对。
- autograd：本接口场景不计算梯度。
- RNG：本文件使用确定输入，不依赖随机流。
- Dyn 维度只检查动态标签，不约束其无语义的 payload；不声明数值算子或 NPU codegen 覆盖。
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
from torch.fx.tensor_type import Dyn, TensorType
from torch.fx.experimental.migrate_gradual_types.constraint import BinConstraintD, BinConstraintT, Conj, DVar, TVar
from torch.fx.experimental.migrate_gradual_types.operation import op_eq
from torch.fx.experimental.migrate_gradual_types.transform_to_z3 import (
    iterate_till_fixed_point, transform_dimension, transform_to_z3, transform_var,
)

from torch.fx.experimental.migrate_gradual_types.constraint_transformation import generate_calc_conv
from torch.fx.experimental.migrate_gradual_types.constraint import CalcConv


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def _tensor_binding(self, identifier, shape):
        value = Dyn if shape == Dyn else TensorType(shape)
        return BinConstraintT(TVar(identifier), value, op_eq)

    def _dimension_bindings(self, values):
        return [BinConstraintD(DVar(identifier), value, op_eq) for identifier, value in values.items()]

    def _formula(self, constraint, counter, bindings):
        normalized, counter = iterate_till_fixed_point(Conj([constraint, *bindings]), counter)
        dimensions = {}
        formula, counter = transform_to_z3(normalized, counter, dimensions)
        return formula, counter, dimensions

    def _check_predicate(self, formula, predicate, details):
        solver = z3.Solver()
        solver.add(formula)
        self.assertEqual(solver.check(), z3.sat, details)
        solver.add(z3.Not(predicate))
        verdict = solver.check()
        self.assertEqual(verdict, z3.unsat, details + (str(solver.model()) if verdict == z3.sat else ""))

    def _sat(self, constraint, counter, bindings, expected):
        formula, _, _ = self._formula(constraint, counter, bindings)
        solver = z3.Solver()
        solver.add(formula)
        self.assertEqual(solver.check(), z3.sat if expected else z3.unsat)

    def _tensor_results(self, constraint, counter, bindings, expected):
        formula, counter, dimensions = self._formula(constraint, counter, bindings)
        predicates = []
        for identifier, shape in expected.items():
            actual, counter = transform_var(TVar(identifier), counter, dimensions)
            if shape == Dyn:
                dynamic, counter = transform_var(Dyn, counter, dimensions)
                predicates.append(actual == dynamic)
                continue
            sort = actual.sort()
            constructor = next(index for index in range(sort.num_constructors())
                               if sort.constructor(index).name() == f"tensor{len(shape)}")
            predicates.append(sort.recognizer(constructor)(actual))
            for index, dimension in enumerate(shape):
                encoded = sort.accessor(constructor, index)(actual)
                dimension_sort = encoded.sort()
                tag = dimension_sort.accessor(0, 0)(encoded)
                value = dimension_sort.accessor(0, 1)(encoded)
                predicates.append(tag == (0 if dimension == Dyn else 1))
                if dimension != Dyn:
                    predicates.append(value == dimension)
        self._check_predicate(formula, z3.And(predicates), f"Expected tensor types {expected}; ")

    def _dimension_results(self, constraint, counter, bindings, expected):
        formula, counter, dimensions = self._formula(constraint, counter, bindings)
        predicates = []
        for identifier, value in expected.items():
            actual, counter = transform_dimension(DVar(identifier), counter, dimensions)
            predicates.append(actual.arg(0) == (0 if value == Dyn else 1))
            if value != Dyn:
                predicates.append(actual.arg(1) == value)
        self._check_predicate(formula, z3.And(predicates), f"Expected dimension types {expected}; ")


    def _run(self, constraint, expected):
        transformed, counter = generate_calc_conv(constraint, 50)
        bindings = [self._tensor_binding(1, [2, 3, 9, 11]),
                    *self._dimension_bindings({3: 2, 4: 3, 5: 9, 6: 11})]
        self._tensor_results(transformed, counter, bindings, {2: expected})

    def test_generate_calc_conv_scalar_kernel_stride_parameters(self):
        dims = [DVar(index) for index in range(3, 7)]
        self._run(CalcConv(TVar(2), TVar(1), 5, 3, 0, 1, 1, dims), [2, 5, 7, 9])

    def test_generate_calc_conv_nonuniform_spatial_parameters(self):
        dims = [DVar(index) for index in range(3, 7)]
        self._run(CalcConv(TVar(2), TVar(1), 5, (3, 3), (1, 2), (2, 3), (1, 2), dims), [2, 5, 5, 4])


if __name__ == "__main__":
    run_tests()
