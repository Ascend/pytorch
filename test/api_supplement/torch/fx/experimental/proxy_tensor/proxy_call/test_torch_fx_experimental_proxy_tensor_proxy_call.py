# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.proxy_tensor.proxy_call 接口功能正确性
API 名称：torch.fx.experimental.proxy_tensor.proxy_call
API 签名：proxy_call(proxy_mode, func, pre_dispatch, args, kwargs) -> object

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 适用的无 mode、空状态或空参数 |
| 枚举选项 | pre_dispatch=False/True |
| 参数类型 | ProxyTorchDispatchMode、OpOverload、Tensor 参数 tuple 和 kwargs 字典 |
| 传参与不传参 | 不适用：五个形参均必传；kwargs 为空或包含 alpha |
| 等价类/边界值 | 一/两输入、alpha、linear 的 pre-dispatch 分支和改变输入 |
| 正常传参场景 | 被测 proxy_call 直接生成 ATen 节点并执行 |
| 异常传参场景 | 未覆盖：无稳定异常契约 |
| 混合设备类型 | 未覆盖：同一加速器输入，不涉及混合设备语义 |

未覆盖项及原因：
- 图执行方式：直接调用被测 tracing/dispatch 入口并执行返回的 FX 图；未覆盖 Inductor codegen。
- eager 对照：相同设备及输入下验证 ATen 图，包含不同输入的重复执行。
- autograd：未覆盖 backward；本文件验证前向捕获。
- RNG：使用确定输入。
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

from torch.fx.experimental import proxy_tensor as pt


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def _tracer(self):
        tracer = pt.PythonKeyTracer()
        tracer.root = torch.nn.Module()
        tracer.graph = torch.fx.Graph()
        return tracer

    def _mode(self):
        return pt.ProxyTorchDispatchMode(self._tracer(), "real")

    def _trace_call(self, op, values, kwargs):
        tracer = self._tracer()
        for index, value in enumerate(values):
            proxy = tracer.create_proxy("placeholder", "value" + str(index), (), {})
            pt.track_tensor(value, proxy, constant=None, tracer=tracer)
        mode = pt.ProxyTorchDispatchMode(tracer, "real")
        result = pt.proxy_call(mode, op, False, values, kwargs)
        slot = pt.fetch_object_proxy(tracer, result)
        self.assertIs(slot.proxy.node.target, op)
        tracer.graph.output(slot.proxy.node)
        return result, torch.fx.GraphModule(tracer.root, tracer.graph)

    def test_proxy_call_direct_functional_dispatch_and_graph_execute_changed_inputs(self):
        self.assertIsNotNone(self.device)
        value = torch.arange(6, dtype=torch.float32, device=self.device).reshape(2, 3)
        result, graph = self._trace_call(torch.ops.aten.neg.default, (value,), {})
        torch.testing.assert_close(result.cpu(), (-value).cpu())
        for current in (value, value + 2):
            actual = graph(current)
            torch.testing.assert_close(actual.cpu(), (-current).cpu())
            self.assertEqual(actual.device.type, self.device_name)
            self.assertEqual(actual.shape, current.shape)
            self.assertEqual(actual.dtype, current.dtype)

    def test_proxy_call_two_input_schema_and_keyword_alpha_preserve_values(self):
        self.assertIsNotNone(self.device)
        first = torch.arange(6, dtype=torch.float32, device=self.device).reshape(2, 3)
        second = first + 1
        result, graph = self._trace_call(torch.ops.aten.add.Tensor, (first, second), {"alpha": 2})
        torch.testing.assert_close(result.cpu(), (first + 2 * second).cpu())
        for shift in (0, 3):
            left, right = first + shift, second - shift
            actual = graph(left, right)
            torch.testing.assert_close(actual.cpu(), (left + 2 * right).cpu())
            self.assertEqual(actual.device.type, self.device_name)


    def test_parameter_true_and_false_predispatch_linear_capture(self):
        value = torch.arange(6, dtype=torch.float32, device=self.device).reshape(2, 3)
        weight = torch.arange(12, dtype=torch.float32, device=self.device).reshape(4, 3) / 10
        bias = torch.arange(4, dtype=torch.float32, device=self.device)
        for pre_dispatch in (False, True):
            with self.subTest(pre_dispatch=pre_dispatch):
                tracer = self._tracer()
                values = (value, weight, bias)
                for index, current in enumerate(values):
                    proxy = tracer.create_proxy("placeholder", "argument" + str(index), (), {})
                    pt.track_tensor(current, proxy, constant=None, tracer=tracer)
                mode = pt.ProxyTorchDispatchMode(tracer, "real", pre_dispatch=pre_dispatch)
                result = pt.proxy_call(mode, torch.ops.aten.linear.default, pre_dispatch, values, {})
                slot = pt.fetch_object_proxy(tracer, result)
                tracer.graph.output(slot.proxy.node)
                graph = torch.fx.GraphModule(tracer.root, tracer.graph)
                for current in (value, value + 1):
                    actual = graph(current, weight, bias)
                    expected = torch.nn.functional.linear(current, weight, bias)
                    torch.testing.assert_close(actual.cpu(), expected.cpu())
                if pre_dispatch:
                    self.assertIs(slot.proxy.node.target, torch.ops.aten.linear.default)


if __name__ == "__main__":
    run_tests()
