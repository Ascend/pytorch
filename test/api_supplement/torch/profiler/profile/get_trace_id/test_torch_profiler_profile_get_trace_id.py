# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.profiler.profile.get_trace_id 接口功能正确性
API 名称：torch.profiler.profile.get_trace_id
API 签名：profile.get_trace_id(self) -> str | None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------------------------------|--------------------------------------------------|
| 空/非空 | 进入前返回 None 与进入后返回非空字符串 | 已覆盖（before_enter_returns_none、inside_context_returns_non_empty_str） |
| 枚举选项 | 枚举参数各合法取值 | 未覆盖，该接口无枚举参数 |
| 参数类型 | 无参方法，多余传参触发 TypeError | 未覆盖，该接口无参数（多余传参由 unexpected_arg_raises 覆盖） |
| 传参与不传参 | 无参正常调用与多余参数调用 | 已覆盖（inside_context_returns_non_empty_str、unexpected_arg_raises） |
| 等价类/边界值 | 进入前/内/后及多会话唯一性 | 已覆盖（before_enter_returns_none、after_exit_retains_trace_id 等） |
| 正常传参场景 | 默认/自定义 ID 及退出后保持 | 已覆盖（inside_context_returns_non_empty_str、custom_trace_id） |
| 异常传参场景 | 传入多余参数触发 TypeError | 已覆盖（unexpected_arg_raises） |
| 混合设备类型 | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖，该接口仅读取 profiler trace_id，不涉及设备迁移 |

未覆盖项及原因：
- 枚举选项：该接口无枚举参数。
- 参数类型：该接口不接受任何参数，非法传参已归入异常传参场景。
- 混合设备类型：该接口仅查询字符串标识，不接受 Tensor 输入。

注意：对可在 CPU/NPU 两侧执行的非图模式浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.profiler import ProfilerActivity

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Probe device only when RUN_FLAG is enabled.
DEVICE_NAME = "cpu"
if RUN_FLAG:
    DEVICE_NAME = None
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = "npu"
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        if not torch.cuda.is_available():
            raise RuntimeError("No available NPU or GPU device.")
        DEVICE_NAME = "cuda"

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestTorchProfilerProfileGetTraceId(TestCase):
    """Test cases for torch.profiler.profile.get_trace_id."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _run_workload(self):
        x = torch.randn(16, 16, device=self.device)
        return (x @ x).sum()

    def test_before_enter_returns_none(self):
        """get_trace_id returns None before entering the profiler context."""
        prof = torch.profiler.profile(activities=[ProfilerActivity.CPU])
        self.assertIsNone(prof.get_trace_id())

    def test_inside_context_returns_non_empty_str(self):
        """get_trace_id returns a non-empty string while profiling is active."""
        with torch.profiler.profile(activities=[ProfilerActivity.CPU]) as prof:
            self._run_workload()
            trace_id = prof.get_trace_id()
            self.assertIsInstance(trace_id, str)
            self.assertGreater(len(trace_id), 0)

    def test_custom_trace_id(self):
        """get_trace_id returns the value produced by custom_trace_id_callback."""
        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU],
            custom_trace_id_callback=lambda: "custom-id-123",
        ) as prof:
            self._run_workload()
            self.assertEqual(prof.get_trace_id(), "custom-id-123")

    def test_after_exit_retains_trace_id(self):
        """The trace ID remains accessible and unchanged after exiting the context."""
        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU],
            custom_trace_id_callback=lambda: "retained-trace-id-456",
        ) as prof:
            self._run_workload()
            active_id = prof.get_trace_id()
            self.assertEqual(active_id, "retained-trace-id-456")

        post_id = prof.get_trace_id()
        self.assertIsInstance(post_id, str)
        self.assertEqual(post_id, active_id)

    def test_distinct_sessions_generate_distinct_default_ids(self):
        """Two separate profiling sessions generate distinct default trace IDs."""
        with torch.profiler.profile(activities=[ProfilerActivity.CPU]) as prof1:
            self._run_workload()
            id1 = prof1.get_trace_id()
        with torch.profiler.profile(activities=[ProfilerActivity.CPU]) as prof2:
            self._run_workload()
            id2 = prof2.get_trace_id()

        self.assertIsInstance(id1, str)
        self.assertIsInstance(id2, str)
        self.assertNotEqual(id1, id2)

    def test_unexpected_arg_raises(self):
        """Passing an unexpected argument to get_trace_id raises TypeError."""
        prof = torch.profiler.profile(activities=[ProfilerActivity.CPU])
        with self.assertRaises(TypeError):
            prof.get_trace_id("unexpected")


if __name__ == "__main__":
    run_tests()
