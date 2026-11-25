# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.profiler.profiler.supported_activities 接口功能正确性
API 名称：torch.profiler.profiler.supported_activities
API 签名：torch.profiler.profiler.supported_activities()

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 返回活动集合非空校验                 | 已覆盖（returns_set_of_profiler_activity）         |
| 枚举选项         | 返回值均为 ProfilerActivity 枚举     | 已覆盖（cpu_always_supported、current_device_activity_supported） |
| 参数类型         | 输入参数类型                         | 未覆盖（见下）                                     |
| 传参与不传参     | 无参调用与多余参数调用               | 已覆盖（returns_set_of_profiler_activity、unexpected_arg_raises） |
| 等价类/边界值    | 多次调用幂等与多后端设备适配         | 已覆盖（idempotent_calls、current_device_activity_supported） |
| 正常传参场景     | 查询活动集合并透传给 profile         | 已覆盖（returns_set_of_profiler_activity、compatible_with_profile 等） |
| 异常传参场景     | 传入多余参数触发 TypeError           | 已覆盖（unexpected_arg_raises）                    |
| 混合设备类型     | 多 Tensor 输入时 NPU/CPU 混合场景    | 未覆盖（见下）                                     |

未覆盖项及原因：
- 参数类型：无参查询接口，不接收外部类型参数。
- 混合设备类型：接口无 Tensor 输入，不涉及多设备混合场景。

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
from torch.profiler.profiler import supported_activities

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
class TestProfilerSupportedActivities(TestCase):
    """Test cases for torch.profiler.profiler.supported_activities."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_returns_set_of_profiler_activity(self):
        """supported_activities() returns a non-empty set of ProfilerActivity members."""
        acts = supported_activities()
        self.assertIsInstance(acts, set)
        self.assertGreater(len(acts), 0)
        for act in acts:
            self.assertIsInstance(act, ProfilerActivity)

    def test_cpu_always_supported(self):
        """ProfilerActivity.CPU is always included in supported_activities()."""
        acts = supported_activities()
        self.assertIn(ProfilerActivity.CPU, acts)

    def test_current_device_activity_supported(self):
        """Current accelerator backend activity is reported when running on accelerator."""
        acts = supported_activities()
        if self.device.type == "cuda":
            self.assertIn(ProfilerActivity.CUDA, acts)
        elif self.device.type != "cpu":
            backend_candidates = [
                getattr(ProfilerActivity, attr)
                for attr in ("PrivateUse1", "NPU", self.device.type.upper())
                if hasattr(ProfilerActivity, attr)
            ]
            self.assertTrue(
                any(act in acts for act in backend_candidates),
                f"Expected one of {backend_candidates} in {acts} for device {self.device.type}",
            )

    def test_idempotent_calls(self):
        """Repeated calls to supported_activities() return equal sets."""
        acts_first = supported_activities()
        acts_second = supported_activities()
        self.assertEqual(acts_first, acts_second)

    def test_compatible_with_profile(self):
        """The returned activity set can be passed directly into torch.profiler.profile."""
        acts = supported_activities()
        x = torch.ones((4, 4), device=self.device)
        with torch.profiler.profile(activities=acts) as prof:
            _ = x + x
        self.assertGreater(len(prof.events()), 0)

    def test_unexpected_arg_raises(self):
        """Passing an unexpected argument to supported_activities() raises TypeError."""
        with self.assertRaises(TypeError):
            supported_activities(1)


if __name__ == "__main__":
    run_tests()
