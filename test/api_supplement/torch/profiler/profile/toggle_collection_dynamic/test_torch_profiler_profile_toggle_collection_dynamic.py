# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.profiler.profile.toggle_collection_dynamic 接口功能正确性
API 名称：torch.profiler.profile.toggle_collection_dynamic
API 签名：profile.toggle_collection_dynamic(enabled, activities)

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 空与非空 activities 列表             | 已覆盖（empty_activities_list、toggle_cpu_off_and_on） |
| 枚举选项         | ProfilerActivity 枚举值              | 已覆盖（toggle_cpu_off_and_on、device_ops_after_reenable） |
| 参数类型         | bool 与活动列表参数类型              | 已覆盖（toggle_cpu_off_and_on、missing_args_raises） |
| 传参与不传参     | 必选参数完整传入与缺失               | 已覆盖（toggle_cpu_off_and_on、missing_args_raises） |
| 等价类/边界值    | 未进入上下文与空活动列表边界         | 已覆盖（before_enter_returns_none、empty_activities_list） |
| 正常传参场景     | 动态启停 CPU 采集与设备算子采集      | 已覆盖（toggle_cpu_off_and_on、device_ops_after_reenable 等） |
| 异常传参场景     | 缺少必选参数触发 TypeError           | 已覆盖（missing_args_raises）                      |
| 混合设备类型     | 多 Tensor 输入时 NPU/CPU 混合场景    | 未覆盖（见下）                                     |

未覆盖项及原因：
- 混合设备类型：接口仅接收布尔开关与活动列表，不涉及多 Tensor 混合设备输入。

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
class TestProfilerProfileToggleCollectionDynamic(TestCase):
    """Test cases for torch.profiler.profile.toggle_collection_dynamic."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_toggle_cpu_off_and_on(self):
        """Disabling and re-enabling CPU collection skips off-window events."""
        x = torch.ones((4, 4))
        with torch.profiler.profile(activities=[ProfilerActivity.CPU]) as prof:
            with torch.profiler.record_function("tag_before"):
                _ = x + 1
            ret_off = prof.toggle_collection_dynamic(False, [ProfilerActivity.CPU])
            self.assertIsNone(ret_off)
            with torch.profiler.record_function("tag_off"):
                _ = x + 2
            ret_on = prof.toggle_collection_dynamic(True, [ProfilerActivity.CPU])
            self.assertIsNone(ret_on)
            with torch.profiler.record_function("tag_on"):
                _ = x + 3

        event_names = {evt.key for evt in prof.events()}
        self.assertIn("tag_before", event_names)
        self.assertIn("tag_on", event_names)
        self.assertNotIn("tag_off", event_names)

    def test_before_enter_returns_none(self):
        """Calling toggle_collection_dynamic before entering context is a no-op returning None."""
        prof = torch.profiler.profile(activities=[ProfilerActivity.CPU])
        self.assertIsNone(prof.profiler)
        ret = prof.toggle_collection_dynamic(False, [ProfilerActivity.CPU])
        self.assertIsNone(ret)

    def test_empty_activities_list(self):
        """Passing an empty activities list leaves active CPU collection unaffected."""
        x = torch.ones((4, 4))
        with torch.profiler.profile(activities=[ProfilerActivity.CPU]) as prof:
            ret = prof.toggle_collection_dynamic(False, [])
            self.assertIsNone(ret)
            with torch.profiler.record_function("tag_after_empty"):
                _ = x * 2

        event_names = {evt.key for evt in prof.events()}
        self.assertIn("tag_after_empty", event_names)

    def test_device_ops_after_reenable(self):
        """Device tensor operations are recorded after re-enabling collection."""
        x = torch.randn((8, 8), device=self.device)
        with torch.profiler.profile(activities=[ProfilerActivity.CPU]) as prof:
            prof.toggle_collection_dynamic(False, [ProfilerActivity.CPU])
            _ = x + x
            prof.toggle_collection_dynamic(True, [ProfilerActivity.CPU])
            with torch.profiler.record_function("device_op_tag"):
                _ = torch.matmul(x, x)

        event_names = {evt.key for evt in prof.events()}
        self.assertIn("device_op_tag", event_names)

    def test_missing_args_raises(self):
        """Omitting required enabled or activities arguments raises TypeError."""
        with torch.profiler.profile(activities=[ProfilerActivity.CPU]) as prof:
            with self.assertRaises(TypeError):
                prof.toggle_collection_dynamic()
            with self.assertRaises(TypeError):
                prof.toggle_collection_dynamic(True)


if __name__ == "__main__":
    run_tests()
