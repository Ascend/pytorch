# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.profiler.profile 接口功能正确性
API 名称：torch.profiler.profile
API 签名：torch.profiler.profile(*, activities=None, schedule=None, on_trace_ready=None,
         record_shapes=False, profile_memory=False, with_stack=False,
         experimental_config=None, with_flops=False, custom_trace_id_callback=None)

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 空 activities 与非空活动集合         | 已覆盖（invalid_args_raise、default_and_explicit_activities） |
| 枚举选项         | ProfilerActivity 枚举组合            | 已覆盖（default_and_explicit_activities）          |
| 参数类型         | 活动集合、回调、配置对象等参数类型   | 已覆盖（default_and_explicit_activities、with_stack_and_flops） |
| 传参与不传参     | 可选开关、schedule、回调传参与默认   | 已覆盖（record_shapes_and_memory、schedule_and_on_trace_ready） |
| 等价类/边界值    | 空活动列表与单周期调度边界           | 已覆盖（invalid_args_raise、schedule_and_on_trace_ready） |
| 正常传参场景     | 各类采集配置正常进入与退出上下文     | 已覆盖（default_and_explicit_activities、with_stack_and_flops 等） |
| 异常传参场景     | 位置参数与空活动列表触发异常         | 已覆盖（invalid_args_raise）                       |
| 混合设备类型     | 多 Tensor 输入时 NPU/CPU 混合场景    | 未覆盖（见下）                                     |

未覆盖项及原因：
- 混合设备类型：torch.profiler.profile 为上下文管理器，不直接接收多 Tensor 输入。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
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
class TestTorchProfilerProfile(TestCase):
    """Test cases for torch.profiler.profile."""

    def setUp(self):
        super().setUp()
        self.device = torch.device(DEVICE_NAME)

    def _run_workload(self):
        x = torch.randn(16, 16, device=self.device)
        return (x @ x).sum()

    def test_default_and_explicit_activities(self):
        """Profiling with supported or explicit CPU activities records events and trace ID."""
        activities = list(torch.profiler.profiler.supported_activities())
        with torch.profiler.profile(activities=activities) as prof:
            self._run_workload()
        averages = prof.key_averages()
        self.assertGreater(len(averages), 0)
        self.assertGreater(sum(evt.count for evt in averages), 0)
        self.assertEqual(prof.activities, set(activities))
        cb = lambda: "custom-profile-001"  # noqa: E731
        with torch.profiler.profile(activities=[ProfilerActivity.CPU], custom_trace_id_callback=cb) as cpu_prof:
            self._run_workload()
            self.assertEqual(cpu_prof.get_trace_id(), "custom-profile-001")
        self.assertGreater(len(cpu_prof.key_averages()), 0)
        self.assertEqual(cpu_prof.activities, {ProfilerActivity.CPU})

    def test_record_shapes_and_memory(self):
        """Enabling record_shapes and profile_memory captures shapes and CPU memory usage."""
        with torch.profiler.profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
            x = torch.randn(64, 64, device="cpu")
            _ = (x + x).sum()
            self._run_workload()
        averages = prof.key_averages(group_by_input_shape=True)
        self.assertTrue(any(len(evt.input_shapes) > 0 for evt in averages))
        self.assertTrue(any(getattr(evt, "cpu_memory_usage", 0) != 0 for evt in averages))

    def test_with_stack_and_flops(self):
        """Enabling with_stack, with_flops, and experimental_config records stacks and FLOPs."""
        exp_cfg = torch._C._profiler._ExperimentalConfig(verbose=True)
        with torch.profiler.profile(activities=[ProfilerActivity.CPU], with_stack=True,
                                    with_flops=True, experimental_config=exp_cfg) as prof:
            a = torch.randn(32, 32, device="cpu")
            _ = torch.mm(a, a)
            self._run_workload()
        averages = prof.key_averages(group_by_stack_n=2)
        self.assertTrue(any(len(evt.stack) > 0 for evt in averages))
        self.assertGreater(sum(evt.flops for evt in averages), 0)

    def test_schedule_and_on_trace_ready(self):
        """Schedule and on_trace_ready invoke the callback at the expected step."""
        recorded_steps = []

        def on_ready(p):
            recorded_steps.append(p.step_num)
            self.assertGreater(len(p.key_averages()), 0)
        sched = torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1)
        with torch.profiler.profile(activities=[ProfilerActivity.CPU], schedule=sched, on_trace_ready=on_ready) as prof:
            for _ in range(4):
                self._run_workload()
                prof.step()
        self.assertEqual(recorded_steps, [3])

    def test_invalid_args_raise(self):
        """Positional activities or an empty activities list raises TypeError or AssertionError."""
        with self.assertRaises(TypeError):
            torch.profiler.profile([ProfilerActivity.CPU])
        with self.assertRaises(AssertionError):
            torch.profiler.profile(activities=[])


if __name__ == "__main__":
    run_tests()
