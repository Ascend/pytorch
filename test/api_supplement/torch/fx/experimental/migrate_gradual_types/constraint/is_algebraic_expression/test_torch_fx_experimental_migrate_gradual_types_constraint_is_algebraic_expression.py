# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.migrate_gradual_types.constraint.is_algebraic_expression 接口功能正确性
API 名称：torch.fx.experimental.migrate_gradual_types.constraint.is_algebraic_expression
API 签名：is_algebraic_expression(constraint) -> bool

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：相关场景中的空维度、空分支或空约束 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：维度变量、静态维度、动态类型或真实回调 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：比较表达式、维度、标量及空 Prod |
| 正常传参场景 | 已覆盖：加减乘除模和 Prod 分类 |
| 异常传参场景 | 不适用：有效约束或索引场景 |
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

from torch.fx.experimental.migrate_gradual_types.constraint import BinConstraintD, DVar, Prod, is_algebraic_expression
from torch.fx.experimental.migrate_gradual_types.operation import op_add, op_sub, op_mul, op_div, op_mod, op_eq


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_is_algebraic_expression_all_arithmetic_operator_kinds_and_product(self):
        for operation in (op_add, op_sub, op_mul, op_div, op_mod):
            with self.subTest(operation=operation):
                self.assertTrue(is_algebraic_expression(BinConstraintD(DVar(1), 3, operation)))
        self.assertTrue(is_algebraic_expression(Prod([DVar(1), 3])))

    def test_is_algebraic_expression_comparison_dimension_literal_and_empty_product(self):
        self.assertFalse(is_algebraic_expression(BinConstraintD(2, 3, op_eq)))
        self.assertFalse(is_algebraic_expression(DVar(1)))
        self.assertFalse(is_algebraic_expression(3))
        self.assertTrue(is_algebraic_expression(Prod([])))


if __name__ == "__main__":
    run_tests()
