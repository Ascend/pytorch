# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.passes.regional_inductor.regional_inductor 接口功能正确性
API 名称：torch.fx.passes.regional_inductor.regional_inductor
API 签名：regional_inductor(gm, *example_args) -> torch.fx.graph_module.GraphModule

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：没有标记区域 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：真实 make_fx 图和 example_args；默认区域配置 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：标记/未标记区域、两组输入及全局函数/配置恢复 |
| 正常传参场景 | 已覆盖：直接 regional_inductor 编译标记区域并执行真实设备结果 |
| 异常传参场景 | 未覆盖：没有稳定的异常契约 |
| 混合设备类型 | 未覆盖：单设备图，无混合设备输入 |

未覆盖项及原因：
- 图执行方式：直接区域编译，计数 wrapper 始终调用真实 standalone_compile，未替换后端能力。
- eager 对照：两组同设备输入，标记区域编译后与原函数对照。
- autograd：未覆盖区域 backward 及序列化的 RegionalOutputCode。
- RNG：确定输入。
- 参数类型：未覆盖嵌套 HOP、异构区域配置和跨 shape 重编译。
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

from unittest.mock import patch
import torch._functorch.config as functorch_config
import torch._inductor
from torch._guards import TracingContext, tracing
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.passes.regional_inductor import regional_inductor


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def _run_region(self, marked):
        self.assertIsNotNone(self.device)
        self.addCleanup(torch._dynamo.reset)
        value = torch.arange(6, dtype=torch.float32, device=self.device).reshape(2, 3) / 5

        def function(x):
            return x.sin() * 2 + 1

        env = ShapeEnv()
        self.addCleanup(env.cleanup)
        fake_mode = FakeTensorMode(shape_env=env)
        with fake_mode:
            fake_value = fake_mode.from_tensor(value, static_shapes=True)
            graph = make_fx(function)(fake_value)
        marked_targets = {torch.ops.aten.sin.default, torch.ops.aten.mul.Tensor}
        if marked:
            for node in graph.graph.nodes:
                if node.target in marked_targets:
                    node.meta["custom"] = {"compile_with_inductor": 0}
        original_compile = torch._inductor.standalone_compile
        compiled_regions = []

        def recording_compile(*args, **kwargs):
            result = original_compile(*args, **kwargs)
            compiled_regions.append(result)
            return result

        original_cache = functorch_config.force_autograd_cache
        original_context = TracingContext.try_get()
        with functorch_config.patch(force_autograd_cache=False):
            with patch.object(torch._inductor, "standalone_compile", recording_compile):
                with tracing(TracingContext(fake_mode)):
                    result = regional_inductor(graph, fake_value)
        self.assertIs(TracingContext.try_get(), original_context)
        self.assertEqual(functorch_config.force_autograd_cache, original_cache)
        self.assertIs(torch._inductor.standalone_compile, original_compile)
        self.assertTrue(callable(result))
        for current in (value, value + 0.4):
            actual = result([current]) if getattr(result, "_boxed_call", False) else result(current)
            expected = function(current)
            torch.testing.assert_close(actual.cpu(), expected.cpu())
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertEqual(actual.device.type, self.device_name)
        return compiled_regions

    def test_regional_inductor_marked_region_calls_real_standalone_inductor_and_matches_eager(self):
        compiled_regions = self._run_region(True)
        self.assertEqual(len(compiled_regions), 1)

    def test_regional_inductor_unmarked_graph_preserves_execution_without_compiling_regions(self):
        self.assertEqual(self._run_region(False), [])


if __name__ == "__main__":
    run_tests()
