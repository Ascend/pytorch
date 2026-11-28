# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.compiler.skip_guard_on_globals_unsafe 接口功能正确性
API 名称：torch.compiler.skip_guard_on_globals_unsafe
API 签名：skip_guard_on_globals_unsafe(guard_entries)

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：空 guard 列表 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：guard 类型/来源；keep_parameters 见适用文件 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：空列表、局部/全局 Tensor、Parameter、内置/用户模块 |
| 正常传参场景 | 已覆盖：真实 GuardFilterEntry 分类及实际 Inductor guard_filter_fn 接入 |
| 异常传参场景 | 未覆盖：这些策略函数没有独立异常参数契约 |
| 混合设备类型 | 未覆盖：同设备输入；本 API 不迁移设备 |

未覆盖项及原因：
- 图执行方式：torch.compile + Inductor，fullgraph=True；本机实际 NPU 后端为 triton_experimental。
- eager 对照：同设备同输入，包含两组不同输入；真实回调记录 guard 类别并检查捕获计数增量。
- autograd：未覆盖 backward；本文件验证 guard 策略前向集成。
- RNG：使用确定输入。
- 参数类型：按 unsafe 接口契约保持模型和全局状态不变；未把删除 guard 当成通用正确性保证。
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
from torch._dynamo.source import (
    GlobalSource, LocalSource, UnspecializedBuiltinNNModuleSource, UnspecializedNNModuleSource,
)
from torch._dynamo.types import GuardFilterEntry
from torch._guards import Guard


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def _entries(self):
        tensor = torch.empty(2, 3, device="meta")
        parameter = torch.nn.Parameter(torch.empty(1, device="meta"))
        specifications = [
            (LocalSource("x"), "TENSOR_MATCH", tensor, False),
            (GlobalSource("global_tensor"), "TENSOR_MATCH", tensor, True),
            (LocalSource("parameter"), "TENSOR_MATCH", parameter, False),
            (GlobalSource("global_state"), "GLOBAL_STATE", None, True),
            (LocalSource("shape_env"), "SHAPE_ENV", None, False),
            (LocalSource("constant"), "EQUALS_MATCH", 3, False),
            (UnspecializedNNModuleSource(LocalSource("model")), "TYPE_MATCH", torch.nn.Identity(), False),
            (UnspecializedBuiltinNNModuleSource(LocalSource("activation")), "TYPE_MATCH", torch.nn.ReLU(), False),
        ]
        return [GuardFilterEntry(name=source.name, has_value=value is not None, value=value,
                                 guard_type=guard_type, derived_guard_types=(), is_global=is_global,
                                 orig_guard=Guard(source, lambda builder, guard: None))
                for source, guard_type, value, is_global in specifications]

    def _compile_with_filter(self, function):
        self.assertIsNotNone(self.device)
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        observed = []

        def filtering(entries):
            observed.append(tuple(entry.guard_type for entry in entries))
            return function(entries)

        def operation(value):
            return value.sin() + value * value

        before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        compiled = torch.compile(operation, backend="inductor", fullgraph=True, dynamic=False,
                                 options={"guard_filter_fn": filtering})
        value = torch.arange(6, dtype=torch.float32, device=self.device).reshape(2, 3) / 4
        for current in (value, value + 0.5):
            actual, expected = compiled(current), operation(current)
            torch.testing.assert_close(actual.cpu(), expected.cpu())
            self.assertEqual(actual.shape, current.shape)
            self.assertEqual(actual.dtype, current.dtype)
            self.assertEqual(actual.device.type, self.device_name)
        self.assertGreater(torch._dynamo.utils.counters["stats"]["unique_graphs"], before)
        self.assertTrue(observed)
        self.assertTrue(any("TENSOR_MATCH" in guards for guards in observed))

    def test_skip_guard_on_globals_unsafe_guard_kinds_sources_and_empty_entries(self):
        entries = self._entries()
        function = torch.compiler.skip_guard_on_globals_unsafe
        self.assertEqual(function(entries), [True, False, True, False, True, True, True, True])
        self.assertEqual(function([]), [])
        self.assertEqual([entry.guard_type for entry in entries],
                         ["TENSOR_MATCH", "TENSOR_MATCH", "TENSOR_MATCH", "GLOBAL_STATE", "SHAPE_ENV",
                          "EQUALS_MATCH", "TYPE_MATCH", "TYPE_MATCH"])

    def test_skip_guard_on_globals_unsafe_actual_guard_filter_option_compiles_and_executes_changed_device_inputs(self):
        self._compile_with_filter(torch.compiler.skip_guard_on_globals_unsafe)


if __name__ == "__main__":
    run_tests()
