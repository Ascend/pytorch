# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.profiler_util.EventList.total_average 接口功能正确性
API 名称：torch.autograd.profiler_util.EventList.total_average
API 签名：EventList.total_average(self) -> FunctionEventAvg

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空列表、单事件与多事件总平均聚合 | 已覆盖（empty_event_list、single_event、aggregates_all_events） |
| 枚举选项 | 无枚举参数 | 未覆盖（无参方法，无枚举选项） |
| 参数类型 | 无参方法传入多余参数报错 | 已覆盖（unexpected_arg_raises） |
| 传参与不传参 | 正常无参调用与传多余参数 | 已覆盖（aggregates_all_events、unexpected_arg_raises） |
| 等价类/边界值 | count=0、单事件、加权 count 平均 | 已覆盖（empty_event_list、single_event、weighted_count_and_avg） |
| 正常传参场景 | 聚合返回总平均 | 已覆盖（empty_event_list、single_event、aggregates_all_events、weighted_count_and_avg） |
| 异常传参场景 | 传入多余位置参数触发 TypeError | 已覆盖（unexpected_arg_raises） |
| 混合设备类型 | 不涉及 Tensor 输入 | 未覆盖（纯 Python 事件统计方法，无 Tensor 输入） |

未覆盖项及原因：
- 枚举选项：接口不接受任何参数，无枚举选项。
- 混合设备类型：接口仅汇总 FunctionEvent 计时属性，不接收 Tensor 参数。

注意：本接口返回聚合后的 FunctionEventAvg 对象，无浮点 Tensor 输出，
     按 API 语义使用返回类型、计数字段、耗时均值及异常断言验证。
"""
import os
import unittest

import torch
from torch.autograd.profiler_util import EventList, FunctionEvent, FunctionEventAvg

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


def _make_event(event_id, name, start_us, end_us):
    return FunctionEvent(
        id=event_id,
        name=name,
        thread=1,
        start_us=start_us,
        end_us=end_us,
    )


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestEventListTotalAverage(TestCase):
    """Test cases for torch.autograd.profiler_util.EventList.total_average."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_empty_event_list(self):
        """total_average() on an empty EventList returns zeroed Total FunctionEventAvg."""
        events = EventList()
        total = events.total_average()
        self.assertIsInstance(total, FunctionEventAvg)
        self.assertEqual(total.key, "Total")
        self.assertEqual(total.count, 0)
        self.assertEqual(total.cpu_time_total, 0)
        self.assertAlmostEqual(total.cpu_time, 0.0, places=6)

    def test_single_event(self):
        """total_average() on a single event matches that event's CPU duration."""
        e1 = _make_event(1, "aten::add", 10, 35)
        total = EventList([e1]).total_average()
        self.assertIsInstance(total, FunctionEventAvg)
        self.assertEqual(total.key, "Total")
        self.assertEqual(total.count, 1)
        self.assertAlmostEqual(total.cpu_time_total, 25.0, places=6)
        self.assertAlmostEqual(total.cpu_time, 25.0, places=6)

    def test_aggregates_all_events(self):
        """total_average() sums totals and computes mean CPU time across all events."""
        e1 = _make_event(1, "aten::add", 0, 10)
        e2 = _make_event(2, "aten::mul", 10, 40)
        e3 = _make_event(3, "aten::relu", 40, 60)
        events = EventList([e1, e2, e3])
        total = events.total_average()
        self.assertIsInstance(total, FunctionEventAvg)
        self.assertEqual(total.key, "Total")
        self.assertEqual(total.count, 3)
        self.assertAlmostEqual(total.cpu_time_total, 60.0, places=6)
        self.assertAlmostEqual(total.self_cpu_time_total, 60.0, places=6)
        self.assertAlmostEqual(total.cpu_time, 20.0, places=6)

    def test_weighted_count_and_avg(self):
        """total_average() weights the mean by each event's count attribute."""
        e1 = _make_event(1, "op_a", 0, 100)
        e1.count = 3
        e2 = _make_event(2, "op_b", 0, 50)
        e2.count = 2
        total = EventList([e1, e2]).total_average()
        self.assertEqual(total.key, "Total")
        self.assertEqual(total.count, 5)
        self.assertAlmostEqual(total.cpu_time_total, 150.0, places=6)
        self.assertAlmostEqual(total.cpu_time, 30.0, places=6)

    def test_unexpected_arg_raises(self):
        """Passing an unexpected argument to total_average() raises TypeError."""
        events = EventList()
        with self.assertRaises(TypeError):
            events.total_average(1)


if __name__ == "__main__":
    run_tests()
