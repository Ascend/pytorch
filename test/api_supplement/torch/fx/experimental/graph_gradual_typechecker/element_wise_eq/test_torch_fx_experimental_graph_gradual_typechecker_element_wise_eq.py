# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.graph_gradual_typechecker.element_wise_eq 接口功能正确性
API 名称：torch.fx.experimental.graph_gradual_typechecker.element_wise_eq
API 签名：element_wise_eq(n) -> list[typing.Any]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 不适用：接口接受 FX Node 或类型注解；未知类型分支按场景覆盖 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：FX Node、TensorType、Dyn 及对应模块参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：非 Node 操作数 |
| 正常传参场景 | 已覆盖：广播后输入维度绑定输出符号 |
| 异常传参场景 | 不适用：不满足前置类型条件时返回空约束 |
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

from torch.fx.experimental.graph_gradual_typechecker import element_wise_eq
from torch.fx.tensor_type import Dyn, TensorType
from torch.fx.experimental.refinement_types import Equality
from sympy import Symbol


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def _node(self, input_type, target=torch.relu, arguments=(), module=False):
        graph = torch.fx.Graph()
        argument = graph.placeholder("x", type_expr=input_type)
        if module:
            node = graph.call_module("layer", (argument, *arguments), {})
        else:
            node = graph.call_function(target, (argument, *arguments), {})
        node.type = Dyn
        graph.output(node)
        return argument, node


    def test_element_wise_eq_broadcasted_dimensions_bind_output_symbols(self):
        graph = torch.fx.Graph()
        first = graph.placeholder("first", type_expr=TensorType((2, 1)))
        second = graph.placeholder("second", type_expr=TensorType((2, 3)))
        node = graph.call_function(torch.add, (first, second))
        batch, feature = Symbol("batch"), Symbol("feature")
        node.type = TensorType((batch, feature))
        self.assertEqual(element_wise_eq(node), [Equality(2, batch), Equality(3, feature)])

    def test_element_wise_eq_non_node_operand_does_not_add_fake_equalities(self):
        _, node = self._node(TensorType((2, 3)), target=torch.add, arguments=(1,))
        node.type = TensorType((2, 3))
        self.assertEqual(element_wise_eq(node), [])


if __name__ == "__main__":
    run_tests()
