# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.profiler.profile.step 接口功能正确性
API 名称：torch.profiler.profile.step
API 签名：profile.step(self) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------------------------------|--------------------------------------------------|
| 空/非空 | 无参步进推进与采集非空事件 | 已覆盖（default_schedule_step_records_ops、schedule_step_increments_and_fires_callback） |
| 枚举选项 | 枚举参数各合法取值 | 未覆盖，step() 本身无枚举参数 |
| 参数类型 | 无参方法，多余传参触发 TypeError | 未覆盖，该接口无参数（多余传参由 unexpected_arg_raises 覆盖） |
| 传参与不传参 | 无参正常步进与传入多余参数 | 已覆盖（default_schedule_step_records_ops、unexpected_arg_raises） |
| 等价类/边界值 | 多轮 repeat 与耗尽后 | 已覆盖（multiple_repeats_triggers_callback_twice、step_after_repeat_exhausted_no_extra_trace） |
| 正常传参场景 | step_num 递增与回调触发 | 已覆盖（schedule_step_increments_and_fires_callback、default_schedule_step_records_ops） |
| 异常传参场景 | 传入多余参数触发 TypeError | 已覆盖（unexpected_arg_raises） |
| 混合设备类型 | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖，该接口为 profiler 状态机推进方法，不接受 Tensor 输入 |

未覆盖项及原因：
- 枚举选项：step() 不含枚举参数。
- 参数类型：该接口不接受任何参数，非法传参已归入异常传参场景。
- 混合设备类型：该接口仅推进 profiler 步数状态机，不接收 Tensor 输入。

注意：对可在 CPU/NPU 两侧执行的非图模式浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.profiler import ProfilerActivity, schedule

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
class TestProfilerProfileStep(TestCase):
    """Test cases for torch.profiler.profile.step."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_schedule_step_increments_and_fires_callback(self):
        """step() advances step_num and triggers on_trace_ready at the scheduled boundary."""
        triggered_steps, captured_counts = [], []

        def on_ready(prof):
            triggered_steps.append(prof.step_num)
            captured_counts.append(len(prof.events()))
        sched = schedule(wait=1, warmup=1, active=2, repeat=1)
        x = torch.ones((4, 4), device=self.device)
        with torch.profiler.profile(activities=[ProfilerActivity.CPU], schedule=sched, on_trace_ready=on_ready) as prof:
            self.assertEqual(prof.step_num, 0)
            for expected_step in range(1, 6):
                _ = x + x
                self.assertIsNone(prof.step())
                self.assertEqual(prof.step_num, expected_step)
        self.assertEqual(triggered_steps, [4])
        self.assertEqual(len(captured_counts), 1)
        self.assertGreater(captured_counts[0], 0)

    def test_default_schedule_step_records_ops(self):
        """step() with the default schedule increments step_num and records op events."""
        x = torch.randn((8, 8), device=self.device)
        with torch.profiler.profile(activities=[ProfilerActivity.CPU]) as prof:
            for expected_step in range(1, 4):
                _ = torch.matmul(x, x)
                prof.step()
                self.assertEqual(prof.step_num, expected_step)
        self.assertGreater(len(prof.events()), 0)

    def test_multiple_repeats_triggers_callback_twice(self):
        """With repeat=2, step() triggers on_trace_ready once per active cycle."""
        triggered_steps = []
        sched = schedule(wait=0, warmup=1, active=1, repeat=2)
        x = torch.ones((4, 4), device=self.device)
        cb = lambda p: triggered_steps.append(p.step_num)  # noqa: E731
        with torch.profiler.profile(activities=[ProfilerActivity.CPU], schedule=sched, on_trace_ready=cb) as prof:
            for _ in range(5):
                _ = x * 2
                prof.step()
        self.assertEqual(triggered_steps, [2, 4])

    def test_step_after_repeat_exhausted_no_extra_trace(self):
        """Extra step() calls after repeat is exhausted do not fire additional callbacks."""
        calls = []
        sched = schedule(wait=0, warmup=1, active=1, repeat=1)
        cb = lambda _p: calls.append(1)  # noqa: E731
        with torch.profiler.profile(activities=[ProfilerActivity.CPU], schedule=sched, on_trace_ready=cb) as prof:
            for _ in range(6):
                _ = torch.zeros((2, 2), device=self.device)
                prof.step()
            self.assertEqual(prof.step_num, 6)
        self.assertEqual(len(calls), 1)

    def test_unexpected_arg_raises(self):
        """Passing an unexpected positional argument to step() raises TypeError."""
        with torch.profiler.profile(activities=[ProfilerActivity.CPU]) as prof:
            with self.assertRaises(TypeError):
                prof.step(1)


if __name__ == "__main__":
    run_tests()
