# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.unify_refinements.convert_eq 接口功能正确性
API 名称：torch.fx.experimental.unify_refinements.convert_eq
API 签名：convert_eq(list_of_eq) -> tuple[tuple[typing.Any, ...], tuple[typing.Any, ...]]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空图、空约束或无依赖状态 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 FX Graph/Node、类型对象及对应接口参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：空等式和重复 lhs |
| 正常传参场景 | 已覆盖：等式左右两侧按原顺序分离 |
| 异常传参场景 | 不适用：有效图和元数据输入 |
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

from torch.fx.tensor_type import TensorType
from torch.fx.experimental.graph_gradual_typechecker import Equality
from torch.fx.experimental.unification.variable import Var
from torch.fx.experimental.unify_refinements import convert_eq


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_convert_eq_lhs_rhs_are_separated_in_original_order(self):
        first, second = Var("first"), Var("second")
        equations = [Equality(first, 3), Equality(second, TensorType((2, 4)))]
        self.assertEqual(convert_eq(equations), ((first, second), (3, TensorType((2, 4)))))
        self.assertEqual(len(equations), 2)

    def test_convert_eq_empty_and_repeated_left_hand_sides(self):
        self.assertEqual(convert_eq([]), ((), ()))
        variable = Var("repeated")
        self.assertEqual(convert_eq([Equality(variable, 2), Equality(variable, 3)]),
                         ((variable, variable), (2, 3)))


if __name__ == "__main__":
    run_tests()
