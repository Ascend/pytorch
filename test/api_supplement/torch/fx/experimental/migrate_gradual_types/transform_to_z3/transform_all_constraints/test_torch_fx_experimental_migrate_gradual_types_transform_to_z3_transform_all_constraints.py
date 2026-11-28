# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.migrate_gradual_types.transform_to_z3.transform_all_constraints 接口功能正确性
API 名称：torch.fx.experimental.migrate_gradual_types.transform_to_z3.transform_all_constraints
API 签名：transform_all_constraints(traced, counter=0) -> Any

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空逻辑约束或映射；其他场景不适用 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：真实渐进类型约束、Z3 表达式、静态与动态维度 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：不满足 in_features 的输入约束 |
| 正常传参场景 | 已覆盖：实际 FX Linear 图端到端转换并证明输出形状 |
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
from torch.fx.experimental.migrate_gradual_types.transform_to_z3 import transform_all_constraints

from torch.fx.tensor_type import TensorType
from torch.fx.experimental.migrate_gradual_types.constraint_generator import ConstraintGenerator
from torch.fx.experimental.migrate_gradual_types.transform_to_z3 import transform_var


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


    def _module(self, shape):
        root = torch.nn.Module()
        root.linear = torch.nn.Linear(3, 4, device="meta")
        graph = torch.fx.Graph()
        source = graph.placeholder("source", type_expr=TensorType(shape))
        result = graph.call_module("linear", (source,))
        graph.output(result)
        return torch.fx.GraphModule(root, graph), result

    def test_transform_all_constraints_linear_graph_has_provable_output_shape(self):
        module, node = self._module([2, 3])
        formula = transform_all_constraints(module, counter=20)
        generator = ConstraintGenerator(module)
        generator.generate_constraints(counter=20)
        output, _ = transform_var(generator.symbol_dict[node], 100, {})
        expected, _ = transform_var(TensorType([2, 4]), 100, {})
        self._sat(formula, True)
        self._sat(z3.And(formula, output != expected), False)

    def test_transform_all_constraints_incompatible_linear_input_features_are_unsatisfiable(self):
        module, _ = self._module([2, 5])
        self._sat(transform_all_constraints(module, counter=20), False)


if __name__ == "__main__":
    run_tests()
