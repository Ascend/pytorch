# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.proxy_tensor.fetch_sym_proxy 接口功能正确性
API 名称：torch.fx.experimental.proxy_tensor.fetch_sym_proxy
API 签名：fetch_sym_proxy(tracer) -> Callable[[PySymType], bool | int | float | Proxy]

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的无 mode、空状态或空参数 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：真实 tracer/mode、Tensor 或符号及默认/显式参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：int/float/bool 常量折叠与一次强制执行 |
| 正常传参场景 | 已覆盖：未绑定符号的真实 thunk proxy 获取 |
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
import sympy
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.experimental.sym_node import SymNode


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

    def test_fetch_sym_proxy_tracked_symbol_forces_lazy_proxy_once(self):
        env = ShapeEnv()
        self.addCleanup(env.cleanup)
        value = env.create_unbacked_symint()
        tracer = self._tracer()
        proxy = tracer.create_proxy("placeholder", "size", (), {})
        calls = []

        def force():
            calls.append("forced")
            return proxy

        pt.set_proxy_slot(value, tracer, pt.Thunk(force))
        getter = pt.fetch_sym_proxy(tracer)
        self.assertIs(getter(value), proxy)
        self.assertIs(getter(value), proxy)
        self.assertEqual(calls, ["forced"])

    def test_fetch_sym_proxy_numeric_symbolic_wrappers_return_python_constants(self):
        env = ShapeEnv()
        self.addCleanup(env.cleanup)
        tracer = self._tracer()
        cases = ((sympy.Integer(4), torch.SymInt, int, 4),
                 (sympy.Float(2.5), torch.SymFloat, float, 2.5),
                 (sympy.true, torch.SymBool, bool, True))
        for expression, wrapper, pytype, hint in cases:
            with self.subTest(pytype=pytype):
                value = wrapper(SymNode(expression, env, pytype, hint, constant=hint if pytype is bool else None))
                result = pt.fetch_sym_proxy(tracer)(value)
                self.assertIsInstance(result, pytype)
                self.assertEqual(result, hint)
        self.assertEqual(list(tracer.graph.nodes), [])


if __name__ == "__main__":
    run_tests()
