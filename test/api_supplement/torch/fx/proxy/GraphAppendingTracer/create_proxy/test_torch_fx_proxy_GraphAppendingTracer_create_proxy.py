# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.proxy.GraphAppendingTracer.create_proxy 接口功能正确性
API 名称：torch.fx.proxy.GraphAppendingTracer.create_proxy
API 签名：GraphAppendingTracer.create_proxy(self, kind, target, args, kwargs, name=None, type_expr=None,
    proxy_factory_fn=None) -> Proxy

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 相关场景中的新图、空字典或空参数 |
| 枚举选项 | placeholder/get_attr/call_module/call_method/call_function/output 六种 kind |
| 参数类型 | callable/字符串 target、tuple args、dict kwargs、name/type_expr 和 Proxy 工厂 |
| 传参与不传参 | name/type_expr/proxy_factory_fn 省略、None/显式值 |
| 等价类/边界值 | 自定义工厂、六种节点图执行和改变输入 |
| 正常传参场景 | 参数降级、指定名字/类型和可执行纯标量图 |
| 异常传参场景 | args 非 tuple 或 kwargs=None 的 AssertionError |
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
import operator


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_create_proxy_proxy_creation_lowers_args_name_and_type(self):
        graph = torch.fx.Graph()
        tracer = GraphAppendingTracer(graph)
        source = tracer.create_proxy("placeholder", "source", (), {})
        result = tracer.create_proxy("call_function", operator.add, (source, 3), {},
                                     name="sum_value", type_expr=int)
        self.assertEqual(result.node.args, (source.node, 3))
        self.assertEqual(result.node.name, "sum_value")
        self.assertIs(result.node.type, int)
        self.assertIn(result.node, source.node.users)
        graph.output(result.node)
        graph.lint()
        self.assertEqual(torch.fx.GraphModule({}, graph)(4), 7)

    def test_create_proxy_custom_proxy_factory_and_non_tuple_arguments(self):
        class SpecializedProxy(torch.fx.Proxy):
            pass

        tracer = GraphAppendingTracer(torch.fx.Graph())
        result = tracer.create_proxy("placeholder", "source", (), {},
                                     proxy_factory_fn=lambda node: SpecializedProxy(node, tracer))
        self.assertIsInstance(result, SpecializedProxy)
        self.assertIs(result.tracer, tracer)
        with self.assertRaisesRegex(AssertionError, "tuple"):
            tracer.create_proxy("call_function", operator.neg, [result], {})


    def test_parameter_all_six_node_kinds_and_explicit_none_options_execute_graph(self):
        root = torch.nn.Module()
        root.activation = torch.nn.ReLU()
        root.bias = 2
        graph = torch.fx.Graph()
        tracer = GraphAppendingTracer(graph)
        value = tracer.create_proxy("placeholder", "value", (), {}, name=None, type_expr=None, proxy_factory_fn=None)
        bias = tracer.create_proxy("get_attr", "bias", (), {})
        activated = tracer.create_proxy("call_module", "activation", (value,), {})
        sine = tracer.create_proxy("call_method", "sin", (activated,), {})
        result = tracer.create_proxy("call_function", torch.add, (sine, bias), {})
        tracer.create_proxy("output", "output", (result,), {})
        self.assertEqual({node.op for node in graph.nodes},
                         {"placeholder", "get_attr", "call_module", "call_method", "call_function", "output"})
        module = torch.fx.GraphModule(root, graph)
        graph.lint()
        for shift in (0.0, 0.5):
            current = torch.arange(-3, 3, device=self.device, dtype=torch.float32) + shift
            torch.testing.assert_close(module(current).cpu(), (current.relu().sin() + 2).cpu())
        with self.assertRaises(AssertionError):
            tracer.create_proxy("call_function", torch.neg, (value,), None)


if __name__ == "__main__":
    run_tests()
