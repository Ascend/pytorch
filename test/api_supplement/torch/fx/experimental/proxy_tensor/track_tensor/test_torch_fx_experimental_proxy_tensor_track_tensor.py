# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.proxy_tensor.track_tensor 接口功能正确性
API 名称：torch.fx.experimental.proxy_tensor.track_tensor
API 签名：track_tensor(tensor, proxy, *, constant, tracer) -> None

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的无 mode、空状态或空参数 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：真实 tracer/mode、Tensor 或符号及默认/显式参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：无常量及延迟 size proxy 强制执行 |
| 正常传参场景 | 已覆盖：Tensor/常量记录、覆盖更新与真实符号 size 绑定 |
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
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv


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

    def test_track_tensor_tensor_record_contains_proxy_and_constant_and_can_be_updated(self):
        tracer = self._tracer()
        tensor = torch.empty(2, 3, device="meta")
        constant = torch.empty_like(tensor)
        first = tracer.create_proxy("placeholder", "first", (), {})
        second = tracer.create_proxy("placeholder", "second", (), {})
        pt.track_tensor(tensor, first, constant=constant, tracer=tracer)
        slot = pt.fetch_object_proxy(tracer, tensor)
        self.assertIs(slot.proxy, first)
        self.assertIs(slot.constant, constant)
        pt.track_tensor(tensor, second, constant=None, tracer=tracer)
        self.assertIs(pt.fetch_object_proxy(tracer, tensor).proxy, second)
        self.assertIsNone(pt.fetch_object_proxy(tracer, tensor).constant)

    def test_track_tensor_symbolic_sizes_receive_lazy_proxy_bindings(self):
        env = ShapeEnv()
        self.addCleanup(env.cleanup)
        mode = FakeTensorMode(shape_env=env)
        tensor = mode.from_tensor(torch.empty(2, 3), static_shapes=False)
        tracer = self._tracer()
        proxy = tracer.create_proxy("placeholder", "value", (), {})
        pt.track_tensor(tensor, proxy, constant=None, tracer=tracer)
        size_proxy = pt.fetch_sym_proxy(tracer)(tensor.shape[0])
        self.assertIsInstance(size_proxy, torch.fx.Proxy)
        self.assertIs(size_proxy.node.target, torch.ops.aten.sym_size.int)
        self.assertIs(size_proxy.node.args[0], proxy.node)
        self.assertEqual(size_proxy.node.args[1], 0)
        self.assertFalse(tracer.enable_thunkify)


if __name__ == "__main__":
    run_tests()
