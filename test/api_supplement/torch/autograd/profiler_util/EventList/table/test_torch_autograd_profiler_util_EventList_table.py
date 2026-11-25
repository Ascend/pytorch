# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.profiler_util.EventList.table 接口功能正确性
API 名称：torch.autograd.profiler_util.EventList.table
API 签名：EventList.table(self, sort_by=None, row_limit=100, max_name_column_width=55, ...)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空 EventList 返回空串与非空格式化表格 | 已覆盖（empty_event_list、default_table_output） |
| 枚举选项 | sort_by 指标、time_unit(s/ms/us)、top_level | 已覆盖（sort_by_and_top_level、time_unit_options） |
| 参数类型 | 字符串表头/排序键、整型行数/列宽、布尔开关 | 已覆盖（sort_by_and_top_level、row_limit_header_and_name_width） |
| 传参与不传参 | 全默认参数与显式格式化选项 | 已覆盖（default_table_output、sort_by_and_top_level、row_limit_header_and_name_width） |
| 等价类/边界值 | 超长名称截断、row_limit=1、多时间单位 | 已覆盖（row_limit_header_and_name_width、time_unit_options） |
| 正常传参场景 | 默认与定制表格输出 | 已覆盖（default_table_output、sort_by_and_top_level、row_limit_header_and_name_width 等） |
| 异常传参场景 | 非法 sort_by 指标触发 AttributeError | 已覆盖（invalid_sort_by_raises） |
| 混合设备类型 | 不涉及多 Tensor 混合设备输入 | 未覆盖（纯字符串表格格式化方法，无 Tensor 输入） |

未覆盖项及原因：
- 混合设备类型：接口仅将 EventList 统计信息格式化为 ASCII 表格字符串，不接收 Tensor 参数。

注意：本接口返回格式化表格字符串，无浮点 Tensor 输出，
     按 API 语义使用返回类型、关键列内容、顺序及异常断言验证。
"""
import os
import unittest

import torch
from torch.autograd.profiler_util import EventList, FunctionEvent

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


def _make_event(event_id, name, start_us=0, end_us=10):
    return FunctionEvent(id=event_id, name=name, thread=1, start_us=start_us, end_us=end_us)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestEventListTable(TestCase):
    """Test cases for torch.autograd.profiler_util.EventList.table."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_empty_event_list(self):
        """Formatting an empty EventList returns an empty string."""
        out = EventList().table()
        self.assertIsInstance(out, str)
        self.assertEqual(out, "")

    def test_default_table_output(self):
        """Default table() output includes standard column headers and event names."""
        events = EventList([_make_event(1, "aten::add", 0, 100), _make_event(2, "aten::mul", 100, 300)])
        out = events.table()
        for token in ("Name", "Self CPU", "CPU total", "# of Calls", "aten::add", "aten::mul"):
            self.assertIn(token, out)

    def test_sort_by_and_top_level(self):
        """sort_by orders rows by metric and top_level_events_only filters child events."""
        parent = _make_event(1, "parent_op", 0, 100)
        child = _make_event(2, "child_op", 10, 90)
        parent.append_cpu_child(child)
        child.set_cpu_parent(parent)
        events = EventList([child, parent])

        by_total = events.table(sort_by="cpu_time_total")
        self.assertLess(by_total.index("parent_op"), by_total.index("child_op"))
        by_self = events.table(sort_by="self_cpu_time_total")
        self.assertLess(by_self.index("child_op"), by_self.index("parent_op"))

        top_only = events.table(top_level_events_only=True)
        self.assertIn("parent_op", top_only)
        self.assertNotIn("child_op", top_only)

    def test_row_limit_header_and_name_width(self):
        """row_limit, header, and max_name_column_width control rows, title, and truncation."""
        long_name = "very_long_operation_name_exceeding_width_limit"
        events = EventList([_make_event(1, long_name, 0, 50), _make_event(2, "second_op", 50, 100)])
        out = events.table(row_limit=1, header="Custom Header", max_name_column_width=15)
        self.assertIn("Custom Header", out)
        self.assertIn(long_name[:12] + "...", out)
        self.assertNotIn(long_name, out)
        self.assertNotIn("second_op", out)

    def test_time_unit_options(self):
        """time_unit formats durations in seconds, milliseconds, or microseconds."""
        events = EventList([_make_event(1, "timed_op", 0, 2000000)])
        self.assertIn("2.000s", events.table(time_unit="s"))
        self.assertIn("2000.000ms", events.table(time_unit="ms"))
        self.assertIn("2000000.000us", events.table(time_unit="us"))

    def test_invalid_sort_by_raises(self):
        """Passing a nonexistent sort_by metric raises AttributeError."""
        events = EventList([_make_event(1, "op_a", 0, 10)])
        with self.assertRaises(AttributeError):
            events.table(sort_by="nonexistent_metric")


if __name__ == "__main__":
    run_tests()
