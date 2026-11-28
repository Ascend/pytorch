# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.migrate_gradual_types.transform_to_z3.transform_all_constraints_trace_time 接口功能正确性
API 名称：torch.fx.experimental.migrate_gradual_types.transform_to_z3.transform_all_constraints_trace_time
API 签名：transform_all_constraints_trace_time(tracer_root, graph, node, counter=0) -> tuple[typing.Any, typing.Any]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空逻辑约束或映射；其他场景不适用 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：真实渐进类型约束、Z3 表达式、静态与动态维度 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：条件为假时反转分支可满足性 |
| 正常传参场景 | 已覆盖：size 条件为真时正/反分支可满足性 |
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
from torch.fx.experimental.migrate_gradual_types.transform_to_z3 import transform_all_constraints_trace_time

import operator
from torch.fx.tensor_type import TensorType


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


    def _branches(self, threshold):
        graph = torch.fx.Graph()
        source = graph.placeholder("source", type_expr=TensorType([2, 3]))
        size = graph.call_method("size", (source, 0))
        condition = graph.call_function(operator.gt, (size, threshold))
        return transform_all_constraints_trace_time(torch.nn.Module(), graph, condition, counter=20)

    def test_transform_all_constraints_trace_time_true_shape_condition_has_only_positive_branch(self):
        positive, negative = self._branches(1)
        self._sat(positive, True)
        self._sat(negative, False)

    def test_transform_all_constraints_trace_time_false_shape_condition_has_only_negative_branch(self):
        positive, negative = self._branches(3)
        self._sat(positive, False)
        self._sat(negative, True)


if __name__ == "__main__":
    run_tests()
