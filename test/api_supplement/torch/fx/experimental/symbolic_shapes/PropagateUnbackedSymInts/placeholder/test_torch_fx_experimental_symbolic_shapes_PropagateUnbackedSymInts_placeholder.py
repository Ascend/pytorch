# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.placeholder 接口功能正确性
API 名称：torch.fx.experimental.symbolic_shapes.PropagateUnbackedSymInts.placeholder
API 签名：PropagateUnbackedSymInts.placeholder(self, target, args, kwargs) -> Any

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空环境、缺失参数或无绑定状态 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际传播解释器、Node/环境及默认/显式参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：变长参数、空余量和缺失必需参数 |
| 正常传参场景 | 已覆盖：位置参数逐次消费及默认参数 |
| 异常传参场景 | 已覆盖：缺失参数 RuntimeError 及 target 类型 AssertionError |
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

import operator
from torch.fx.experimental.symbolic_shapes import PropagateUnbackedSymInts


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def _fixture(self):
        class Scale(torch.nn.Module):
            def forward(self, value, *, factor=3):
                return value * factor

        root = torch.nn.Module()
        root.block = torch.nn.Module()
        root.block.scale = Scale()
        root.block.bias = 7
        graph = torch.fx.Graph()
        source = graph.placeholder("value")
        call = graph.call_module("block.scale", (source,), {"factor": 3})
        bias = graph.create_node("get_attr", "block.bias", (), {})
        result = graph.call_function(operator.add, (call, bias))
        graph.output({"result": result, "input": source})
        module = torch.fx.GraphModule(root, graph)
        return PropagateUnbackedSymInts(module), source, call, bias

    def test_placeholder_positional_arguments_are_consumed_and_default_is_used(self):
        interpreter, _, _, _ = self._fixture()
        interpreter.args_iter = iter((2, 5))
        self.assertEqual(interpreter.placeholder("first", (), {}), 2)
        self.assertEqual(interpreter.placeholder("second", (), {}), 5)
        self.assertEqual(interpreter.placeholder("default", (11,), {}), 11)
        with self.assertRaisesRegex(RuntimeError, "Expected positional argument"):
            interpreter.placeholder("required", (), {})

    def test_placeholder_starred_arguments_consume_remaining_values_and_target_type_is_checked(self):
        interpreter, _, _, _ = self._fixture()
        interpreter.args_iter = iter((3, 4, 5))
        self.assertEqual(interpreter.placeholder("first", (), {}), 3)
        self.assertEqual(interpreter.placeholder("*rest", (), {}), [4, 5])
        self.assertEqual(interpreter.placeholder("*empty", (), {}), [])
        with self.assertRaisesRegex(AssertionError, "Expected target to be str"):
            interpreter.placeholder(3, (), {})


if __name__ == "__main__":
    run_tests()
