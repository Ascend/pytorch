# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.proxy_tensor.dispatch_trace 接口功能正确性
API 名称：torch.fx.experimental.proxy_tensor.dispatch_trace
API 签名：dispatch_trace(root, tracer, concrete_args=None) -> GraphModule

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的无 mode、空状态或空参数 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：真实 tracer/mode、Tensor 或符号及默认/显式参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：无参数 root 与嵌套常量输出 |
| 正常传参场景 | 已覆盖：直接 tracing 入口生成可执行 GraphModule |
| 异常传参场景 | 未覆盖：无稳定异常契约 |
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

from torch.fx.experimental import proxy_tensor as pt


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_dispatch_trace_direct_trace_entry_returns_executable_scalar_graph(self):
        def function(value):
            return value * 2 + 1

        module = pt.dispatch_trace(function, torch.fx.Tracer())
        self.assertIsInstance(module, torch.fx.GraphModule)
        self.assertEqual(module(3), 7)
        self.assertEqual(module(-2), -3)
        self.assertTrue(any(node.op == "call_function" for node in module.graph.nodes))
        module.graph.lint()

    def test_dispatch_trace_zero_argument_root_and_explicit_none_concrete_args(self):
        def function():
            return {"value": 7, "other": (None, 0)}

        module = pt.dispatch_trace(function, torch.fx.Tracer(), concrete_args=None)
        self.assertEqual(module(), {"value": 7, "other": (None, 0)})
        self.assertFalse(any(node.op == "placeholder" for node in module.graph.nodes))


if __name__ == "__main__":
    run_tests()
