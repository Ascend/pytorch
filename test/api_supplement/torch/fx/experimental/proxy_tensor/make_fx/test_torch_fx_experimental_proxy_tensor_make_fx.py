# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.proxy_tensor.make_fx 接口功能正确性
API 名称：torch.fx.experimental.proxy_tensor.make_fx
API 签名：make_fx(f, decomposition_table=None, tracing_mode='real', _allow_non_fake_inputs=False, *, pre_dispatch=False,
    record_module_stack=False, _allow_fake_constant=False, _error_on_data_dependent_ops=True,
    record_stack_traces=False, proxy_module_inputs=False, _disable_torch_fn_metadata_mode=False, dynamic_shapes=None)
    -> Callable[..., GraphModule]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的无 mode、空状态或空参数 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：真实 tracer/mode、Tensor 或符号及默认/显式参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：非连续输入、不同输入重复执行及非法 mode |
| 正常传参场景 | 已覆盖：真实与 fake ATen 图捕获、结构及设备数值 |
| 异常传参场景 | 已覆盖：非法 tracing_mode AssertionError |
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

    def _compare(self, graph, function, value):
        for current in (value, value + 0.4):
            actual, expected = graph(current), function(current)
            self.assertEqual(actual.keys(), expected.keys())
            for name in expected:
                torch.testing.assert_close(actual[name].cpu(), expected[name].cpu())
                self.assertEqual(actual[name].device.type, self.device_name)
                self.assertEqual(actual[name].shape, expected[name].shape)
                self.assertEqual(actual[name].dtype, expected[name].dtype)

    def test_make_fx_real_tracing_captures_aten_graph_and_executes_changed_inputs(self):
        self.assertIsNotNone(self.device)
        value = torch.arange(6, dtype=torch.float32, device=self.device).reshape(2, 3) / 4

        def function(x):
            return {"value": x.sin() + 2, "view": x.transpose(0, 1)}

        graph = pt.make_fx(function, record_stack_traces=True)(value)
        self.assertTrue(any(node.target is torch.ops.aten.sin.default for node in graph.graph.nodes))
        graph.graph.lint()
        self._compare(graph, function, value)

    def test_make_fx_fake_capture_executes_real_device_inputs_and_invalid_mode_is_rejected(self):
        self.assertIsNotNone(self.device)
        value = torch.arange(6, dtype=torch.float32, device=self.device).reshape(3, 2).transpose(0, 1)

        def function(x):
            return {"value": x.cos() * 3, "view": x.transpose(0, 1)}

        graph = pt.make_fx(function, tracing_mode="fake", decomposition_table={})(value)
        self._compare(graph, function, value)
        with self.assertRaisesRegex(AssertionError, "tracing_mode must be"):
            pt.make_fx(function, tracing_mode="invalid")


if __name__ == "__main__":
    run_tests()
