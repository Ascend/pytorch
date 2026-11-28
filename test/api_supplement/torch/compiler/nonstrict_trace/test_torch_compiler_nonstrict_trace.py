# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.compiler.nonstrict_trace 接口功能正确性
API 名称：torch.compiler.nonstrict_trace
API 签名：nonstrict_trace(traceable_fn) -> collections.abc.Callable[~_P, ~_R]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 未覆盖：本批使用非空小张量 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际装饰器、默认/显式参数及加速器 Tensor |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：两组输入及显式 Tensor 参数的 backward |
| 正常传参场景 | 已覆盖：nonstrict 区域、内部 Dynamo graph_break 与 fullgraph 外层 |
| 异常传参场景 | 未覆盖：没有稳定的异常契约 |
| 混合设备类型 | 未覆盖：单设备小图，无混合设备语义 |

未覆盖项及原因：
- 图执行方式：装饰目标函数后用 Inductor/fullgraph 编译；记录实际 GraphModule，重复执行只捕获一个图。
- eager 对照：两组不同的同设备输入；wrap_numpy 使用等价 Torch 语义作为基线。
- autograd：已覆盖：两组输入的前向和 backward 梯度。
- RNG：使用确定输入。
- 参数类型：未覆盖用户 pytree 类、Module 参数和闭包 Tensor 常量。
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

import torch._dynamo
from torch._dynamo.backends.registry import lookup_backend


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def _compile_function(self, function):
        self.assertIsNotNone(self.device)
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        captured = []

        def backend(module, example_inputs):
            captured.append(module)
            return lookup_backend("inductor")(module, example_inputs)

        compiled = torch.compile(function, backend=backend, fullgraph=True, dynamic=False)
        return compiled, captured

    def _assert_forward(self, function, golden):
        compiled, captured = self._compile_function(function)
        value = torch.arange(6, dtype=torch.float32, device=self.device).reshape(2, 3) / 5
        for current in (value, value + 0.4):
            actual, expected = compiled(current), golden(current)
            torch.testing.assert_close(actual.cpu(), expected.cpu())
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertEqual(actual.device.type, self.device_name)
        self.assertEqual(len(captured), 1)
        return captured[0]

    def _assert_backward(self, function, golden):
        compiled, captured = self._compile_function(function)
        base = torch.arange(6, dtype=torch.float32, device=self.device).reshape(2, 3) / 5
        for shift in (0.0, 0.4):
            first = (base + shift).detach().requires_grad_()
            second = (base + shift).detach().requires_grad_()
            actual, expected = compiled(first), golden(second)
            actual.sum().backward()
            expected.sum().backward()
            torch.testing.assert_close(actual.cpu(), expected.cpu())
            torch.testing.assert_close(first.grad.cpu(), second.grad.cpu())
            self.assertEqual(actual.device.type, self.device_name)
        self.assertEqual(len(captured), 1)

    def test_nonstrict_trace_nonstrict_region_with_internal_graph_break_compiles_fullgraph(self):
        @torch.compiler.nonstrict_trace
        def region(value):
            torch._dynamo.graph_break()
            return value.sin() + 1

        def function(value):
            return region(value) * 2

        self._assert_forward(function, lambda x: (x.sin() + 1) * 2)

    def test_nonstrict_trace_explicit_tensor_inputs_preserve_gradients(self):
        @torch.compiler.nonstrict_trace
        def region(value):
            return value.sin() + value * value

        self._assert_backward(region, lambda x: x.sin() + x * x)


if __name__ == "__main__":
    run_tests()
