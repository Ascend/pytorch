# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.migrate_gradual_types.constraint_generator.gen_embedding_rules 接口功能正确性
API 名称：torch.fx.experimental.migrate_gradual_types.constraint_generator.gen_embedding_rules
API 签名：gen_embedding_rules(n, symbols, embedding_dim, counter) ->
    tuple[list[torch.fx.experimental.migrate_gradual_types.constraint.Constraint], int]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 非空和零尺寸索引 Tensor |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | FX Node、整数/DVar embedding_dim、符号表和计数器 |
| 传参与不传参 | 不适用：接口中的 Node、类型/符号状态及计数器形参均必传 |
| 等价类/边界值 | 零尺寸索引和 DVar/整数等价绑定 |
| 正常传参场景 | 整数和符号 embedding_dim 绑定的唯一形状 |
| 异常传参场景 | 未覆盖：本文件证明有效输入的约束及唯一输出形状，不推测参数注解以外的异常验证 |
| 混合设备类型 | 不适用：本文件为设备无关的图结构、类型或辅助状态语义 |

未覆盖项及原因：
- 图执行方式：本文件直接验证图模式辅助接口，不声明 NPU Inductor codegen 覆盖。
- eager 对照：使用接口定义的结构、状态或确定值；不构造无意义的 CPU/NPU 自比对。
- autograd：本接口场景不计算梯度。
- RNG：本文件使用确定输入，不依赖随机流。
- Z3 同时证明约束可满足及输出不可能为其他形状；不声明数值算子或 NPU codegen 覆盖。
- 未覆盖：标量索引和输出 rank 超过 4；当前 TensorType 的 Z3 编码仅支持 rank 1 至 4。
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
from torch.fx.tensor_type import TensorType
from torch.fx.experimental.migrate_gradual_types.constraint import BinConstraintT, Conj, TVar
from torch.fx.experimental.migrate_gradual_types.operation import op_eq
from torch.fx.experimental.migrate_gradual_types.transform_to_z3 import (
    iterate_till_fixed_point, transform_to_z3, transform_var,
)

from torch.fx.experimental.migrate_gradual_types.constraint_generator import gen_embedding_rules


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def _inputs(self, shapes):
        graph = torch.fx.Graph()
        nodes = [graph.placeholder(f"input_{index}") for index in range(len(shapes))]
        symbols = {node: TVar(index + 1) for index, node in enumerate(nodes)}
        return graph, nodes, symbols

    def _prove_shape(self, nodes, shapes, symbols, output, constraints, counter, expected):
        bindings = [BinConstraintT(symbols[node], TensorType(shape), op_eq) for node, shape in zip(nodes, shapes)]
        normalized, counter = iterate_till_fixed_point(Conj([*constraints, *bindings]), counter)
        dimension_map = {}
        formula, counter = transform_to_z3(normalized, counter, dimension_map)
        actual, counter = transform_var(symbols[output], counter, dimension_map)
        encoded_expected, _ = transform_var(TensorType(expected), counter, dimension_map)
        solver = z3.Solver()
        solver.add(formula)
        self.assertEqual(solver.check(), z3.sat, f"Valid input shapes {shapes} should allow output {expected}")
        solver.add(actual != encoded_expected)
        verdict = solver.check()
        details = (f"Expected shape {expected}; counterexample {solver.model().eval(actual, model_completion=True)}"
                   if verdict == z3.sat else solver.reason_unknown() if verdict == z3.unknown else "")
        self.assertEqual(verdict, z3.unsat, details)


    def _run(self, shape, expected):
        shapes = [shape]
        graph, nodes, symbols = self._inputs(shapes)
        node = graph.call_module("embedding", (nodes[0],))
        constraints, counter = gen_embedding_rules(node, symbols, 4, 50)
        self._prove_shape(nodes, shapes, symbols, node, constraints, counter, expected)

    def test_gen_embedding_rules_embedding_dimension_is_appended(self):
        self._run([2, 3], [2, 3, 4])

    def test_gen_embedding_rules_empty_index_tensor_keeps_embedding_dimension(self):
        self._run([0, 2], [0, 2, 4])


    def test_parameter_symbolic_feature_dimensions_match_integer_form(self):
        from torch.fx.experimental.migrate_gradual_types.constraint import BinConstraintD, DVar

        shape = [2, 3]
        graph, nodes, symbols = self._inputs([shape])
        node = graph.call_module("layer", (nodes[0],))
        features = DVar(2)
        constraints, counter = gen_embedding_rules(node, symbols, features, 50)
        constraints.append(BinConstraintD(features, 4, op_eq))
        self._prove_shape(nodes, [shape], symbols, node, constraints, counter, [2, 3, 4])


if __name__ == "__main__":
    run_tests()
