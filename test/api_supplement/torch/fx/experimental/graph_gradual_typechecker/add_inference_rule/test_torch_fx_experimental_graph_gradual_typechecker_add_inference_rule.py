# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.graph_gradual_typechecker.add_inference_rule 接口功能正确性
API 名称：torch.fx.experimental.graph_gradual_typechecker.add_inference_rule
API 签名：add_inference_rule(n) -> Any

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 不适用：接口接受 FX Node 或类型注解；未知类型分支按场景覆盖 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：FX Node、TensorType、Dyn 及对应模块参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：标量/张量和不兼容维度 |
| 正常传参场景 | 已覆盖：广播形状、节点状态和原输入保持 |
| 异常传参场景 | 已覆盖：不兼容的类型或参数 |
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

from torch.fx.experimental.graph_gradual_typechecker import add_inference_rule
from torch.fx.tensor_type import Dyn, TensorType


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_add_inference_rule_broadcast_type_and_operand_preservation(self):
        graph = torch.fx.Graph()
        first = graph.placeholder("first", type_expr=TensorType((2, 1, 4)))
        second = graph.placeholder("second", type_expr=TensorType((1, 3, 4)))
        node = graph.call_function(torch.add, (first, second))
        node.type = Dyn
        self.assertEqual(add_inference_rule(node), TensorType((2, 3, 4)))
        self.assertEqual(node.type, TensorType((2, 3, 4)))
        self.assertEqual(first.type, TensorType((2, 1, 4)))
        self.assertEqual(second.type, TensorType((1, 3, 4)))
        self.assertTrue(node.meta["broadcast"])

    def test_add_inference_rule_scalar_tensor_and_incompatible_types(self):
        graph = torch.fx.Graph()
        scalar = graph.placeholder("scalar", type_expr=int)
        tensor = graph.placeholder("tensor", type_expr=TensorType((2, 3)))
        node = graph.call_function(torch.add, (scalar, tensor))
        node.type = Dyn
        self.assertEqual(add_inference_rule(node), TensorType((2, 3)))
        other = graph.placeholder("other", type_expr=TensorType((5, 7)))
        bad = graph.call_function(torch.add, (tensor, other))
        bad.type = Dyn
        with self.assertRaises(TypeError):
            add_inference_rule(bad)


if __name__ == "__main__":
    run_tests()
