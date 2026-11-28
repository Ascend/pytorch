# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.proxy.GraphAppendingTracer.create_arg 接口功能正确性
API 名称：torch.fx.proxy.GraphAppendingTracer.create_arg
API 签名：GraphAppendingTracer.create_arg(self, a) -> Union[tuple['Argument', ...], collections.abc.Sequence['Argument'],
    collections.abc.Mapping[str, 'Argument'], slice, range, ForwardRef('Node'), str, int, float, bool, complex,
    torch.dtype, torch.Tensor, torch.device, torch.memory_format, torch.layout, torch._ops.OpOverload, torch.SymInt,
    torch.SymBool, torch.SymFloat, NoneType]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | None、空容器/字符串/range 和 dataclass 的 None 字段 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | Proxy、常量、Enum、range/slice、NamedTuple/dataclass、自定义协议及 dtype/device 等 |
| 传参与不传参 | 不适用：a 必传 |
| 等价类/边界值 | 嵌套字段、正/负步长 range、dataclass 图执行和不支持对象 |
| 正常传参场景 | 嵌套 Proxy、slice、None 转换为真实 IR 参数 |
| 异常传参场景 | 无协议的不支持对象抛出 NotImplementedError |
| 混合设备类型 | 不适用：本文件为设备无关的图结构、类型或辅助状态语义 |

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

from torch.fx.proxy import GraphAppendingTracer
from collections import namedtuple


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_create_arg_nested_proxy_arguments_are_lowered_to_nodes(self):
        graph = torch.fx.Graph()
        tracer = GraphAppendingTracer(graph)
        node = graph.placeholder("value")
        proxy = tracer.proxy(node)
        result = tracer.create_arg({"value": [proxy, (7,)], "slice": slice(proxy, 9, 2), "none": None})
        self.assertEqual(result, {"value": [node, (7,)], "slice": slice(node, 9, 2), "none": None})
        self.assertEqual(tracer.create_arg([]), [])

    def test_create_arg_namedtuple_custom_protocol_and_unsupported_object(self):
        tracer = GraphAppendingTracer(torch.fx.Graph())
        pair_type = namedtuple("Pair", ["left", "right"])
        proxy = tracer.create_proxy("placeholder", "value", (), {})
        result = tracer.create_arg(pair_type(proxy, 3))
        self.assertIsInstance(result, pair_type)
        self.assertEqual(result, pair_type(proxy.node, 3))

        class CustomArgument:
            def __fx_create_arg__(self, current_tracer):
                return current_tracer.create_arg((proxy, 11))

        self.assertEqual(tracer.create_arg(CustomArgument()), (proxy.node, 11))
        with self.assertRaisesRegex(NotImplementedError, "argument of type"):
            tracer.create_arg(object())


    def test_parameter_literal_enum_range_and_dataclass_protocols(self):
        from dataclasses import dataclass
        from enum import Enum

        class Choice(Enum):
            FIRST = 1

        tracer = GraphAppendingTracer(torch.fx.Graph())
        for value in (None, Ellipsis, False, 0, -1, 2.5, 1 + 2j, "", torch.float32,
                      self.device, torch.contiguous_format, torch.strided, torch.ops.aten.neg.default, Choice.FIRST):
            self.assertIs(tracer.create_arg(value), value)
        for value in (range(0), range(1, 7, 2), range(5, 0, -1)):
            self.assertEqual(tracer.create_arg(value), value)

        @dataclass
        class Pair:
            left: object
            right: object

        proxy = tracer.create_proxy("placeholder", "value", (), {})
        node = tracer.create_arg(Pair(proxy, None))
        self.assertEqual(node.op, "call_function")
        self.assertIs(node.target, Pair)
        self.assertEqual(node.kwargs, {"left": proxy.node, "right": None})
        tracer.graph.output(node)
        module = torch.fx.GraphModule({}, tracer.graph)
        result = module(7)
        self.assertIs(type(result), Pair)
        self.assertEqual(result.left, 7)
        self.assertIsNone(result.right)


if __name__ == "__main__":
    run_tests()
