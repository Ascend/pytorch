# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.profiler.profile.set_custom_trace_id_callback 接口功能正确性
API 名称：torch.profiler.profile.set_custom_trace_id_callback
API 签名：profile.set_custom_trace_id_callback(self, callback) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------------------------------|--------------------------------------------------|
| 空/非空 | callback 为 None 与可调用对象 | 已覆盖（set_none_falls_back_to_default、set_callback_before_enter） |
| 枚举选项 | 枚举参数各合法取值 | 未覆盖，该接口无枚举参数 |
| 参数类型 | 函数、lambda 回调与 None | 已覆盖（set_callback_before_enter、set_none_falls_back_to_default） |
| 传参与不传参 | 传入 callback 与缺参异常 | 已覆盖（set_callback_before_enter、missing_arg_raises） |
| 等价类/边界值 | 进入前设置、跨周期更新与置 None | 已覆盖（update_callback_across_schedule_cycles、set_none_falls_back_to_default） |
| 正常传参场景 | 自定义回调生效与回退默认 ID | 已覆盖（set_callback_before_enter、update_callback_across_schedule_cycles） |
| 异常传参场景 | 缺少必填参数触发 TypeError | 已覆盖（missing_arg_raises） |
| 混合设备类型 | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖，该接口仅注册 trace_id 回调函数，不涉及设备迁移 |

未覆盖项及原因：
- 枚举选项：该接口无枚举参数。
- 混合设备类型：该接口仅注册回调函数或 None，不接受 Tensor 输入。

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
class TestTorchProfilerProfileSetCustomTraceIdCallback(TestCase):
    """Test cases for torch.profiler.profile.set_custom_trace_id_callback."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _run_workload(self):
        x = torch.randn(16, 16, device=self.device)
        return (x @ x).sum()

    def test_set_callback_before_enter(self):
        """Setting a callback before entering supplies the active session trace ID."""
        call_count = [0]

        def make_id():
            call_count[0] += 1
            return "set-cb-trace-001"
        prof = torch.profiler.profile(activities=[ProfilerActivity.CPU])
        ret = prof.set_custom_trace_id_callback(make_id)
        self.assertIsNone(ret)
        self.assertIs(prof.custom_trace_id_callback, make_id)
        with prof:
            self._run_workload()
            self.assertEqual(prof.get_trace_id(), "set-cb-trace-001")
        self.assertEqual(call_count[0], 1)

    def test_update_callback_across_schedule_cycles(self):
        """Updating the callback between schedule cycles changes subsequent trace IDs."""
        observed_ids = []

        def on_ready(p):
            observed_ids.append(p.get_trace_id())
            if len(observed_ids) == 1:
                p.set_custom_trace_id_callback(lambda: "cycle-2-trace-id")
        sched = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=2)
        prof = torch.profiler.profile(activities=[ProfilerActivity.CPU], schedule=sched, on_trace_ready=on_ready)
        prof.set_custom_trace_id_callback(lambda: "cycle-1-trace-id")
        with prof:
            for _ in range(4):
                self._run_workload()
                prof.step()
        self.assertEqual(observed_ids, ["cycle-1-trace-id", "cycle-2-trace-id"])

    def test_set_none_falls_back_to_default(self):
        """Passing None clears the custom callback and falls back to default trace IDs."""
        prof = torch.profiler.profile(
            activities=[ProfilerActivity.CPU],
            custom_trace_id_callback=lambda: "initial-custom-id",
        )
        prof.set_custom_trace_id_callback(None)
        self.assertIsNone(prof.custom_trace_id_callback)
        with prof:
            self._run_workload()
            trace_id = prof.get_trace_id()
            self.assertIsInstance(trace_id, str)
            self.assertGreater(len(trace_id), 0)
            self.assertNotEqual(trace_id, "initial-custom-id")

    def test_missing_arg_raises(self):
        """Calling set_custom_trace_id_callback with no argument raises TypeError."""
        prof = torch.profiler.profile(activities=[ProfilerActivity.CPU])
        with self.assertRaises(TypeError):
            prof.set_custom_trace_id_callback()


if __name__ == "__main__":
    run_tests()
