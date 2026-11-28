# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.migrate_gradual_types.transform_to_z3.transform_dimension 接口功能正确性
API 名称：torch.fx.experimental.migrate_gradual_types.transform_to_z3.transform_dimension
API 签名：transform_dimension(dimension, counter, dimension_dict) -> tuple[typing.Any, int]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空逻辑约束或映射；其他场景不适用 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：真实渐进类型约束、Z3 表达式、静态与动态维度 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：fresh tag 计数与矛盾变量赋值 |
| 正常传参场景 | 已覆盖：静态/动态维度编码和变量重复复用 |
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
from torch.fx.experimental.migrate_gradual_types.transform_to_z3 import transform_dimension

from torch.fx.tensor_type import Dyn
from torch.fx.experimental.migrate_gradual_types.constraint import DVar
from torch.fx.experimental.migrate_gradual_types.transform_to_z3 import D


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


    def test_transform_dimension_static_and_dynamic_dimension_tags(self):
        static, counter = transform_dimension(7, 20, {})
        self.assertEqual(static.arg(0).as_long(), 1)
        self.assertEqual(static.arg(1).as_long(), 7)
        self.assertEqual(counter, 20)
        dynamic, counter = transform_dimension(Dyn, counter, {})
        self.assertEqual(dynamic.arg(0).as_long(), 0)
        self.assertEqual(counter, 21)
        self._sat(dynamic == D(1, 7), False)

    def test_transform_dimension_repeated_dimension_variable_reuses_encoding_and_counter(self):
        dimensions = {}
        first, counter = transform_dimension(DVar(3), 20, dimensions)
        again, later = transform_dimension(DVar(3), counter, dimensions)
        self.assertTrue(first.eq(again))
        self.assertEqual(counter, 21)
        self.assertEqual(later, counter)
        self._sat(z3.And(first == D(1, 7), again == D(1, 8)), False)
        self._sat(first == D(1, 7), True)


if __name__ == "__main__":
    run_tests()
