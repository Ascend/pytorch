# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.profiler_util.EventList.sort 接口功能正确性
API 名称：torch.autograd.profiler_util.EventList.sort
API 签名：EventList.sort(self, *, key=None, reverse=False) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空列表、单元素与多元素排序 | 已覆盖（sort_empty_and_single、sort_by_cpu_time_asc_desc） |
| 枚举选项 | reverse=False/True 升序与降序 | 已覆盖（sort_by_cpu_time_asc_desc） |
| 参数类型 | 关键字 key 函数与位置参数报错 | 已覆盖（sort_by_cpu_time_asc_desc、positional_key_raises） |
| 传参与不传参 | 默认排序与显式 key/reverse | 已覆盖（sort_empty_and_single、sort_by_cpu_time_asc_desc、unorderable_default_key_raises） |
| 等价类/边界值 | 相同键值排序稳定性、空/单元素边界 | 已覆盖（sort_by_name_and_stability、sort_empty_and_single） |
| 正常传参场景 | 按耗时升降序、按名称排序及稳定性验证 | 已覆盖（sort_by_cpu_time_asc_desc、sort_by_name_and_stability、sort_empty_and_single） |
| 异常传参场景 | 无 key 不可比较对象与位置参数 TypeError | 已覆盖（unorderable_default_key_raises、positional_key_raises） |
| 混合设备类型 | 不涉及 Tensor 输入 | 未覆盖（纯 Python 容器方法，无 Tensor 输入） |

未覆盖项及原因：
- 混合设备类型：接口为 list 子类原地排序方法，不接收 Tensor 参数。

注意：本接口为 EventList 容器原地排序方法，无浮点 Tensor 输出，
     按 API 语义使用返回值 None、元素顺序引用及异常断言验证。
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
class TestEventListSort(TestCase):
    """Test cases for torch.autograd.profiler_util.EventList.sort."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_sort_by_cpu_time_asc_desc(self):
        """Sorting by cpu_time_total orders events ascending and descending in place."""
        e_mid, e_fast, e_slow = _make_event(1, "m", 0, 20), _make_event(2, "f", 0, 5), _make_event(3, "s", 0, 50)
        events = EventList([e_mid, e_fast, e_slow])
        alias = events

        self.assertIsNone(events.sort(key=lambda e: e.cpu_time_total))
        self.assertIs(events, alias)
        for idx, expected in enumerate([e_fast, e_mid, e_slow]):
            self.assertIs(events[idx], expected)

        self.assertIsNone(events.sort(key=lambda e: e.cpu_time_total, reverse=True))
        for idx, expected in enumerate([e_slow, e_mid, e_fast]):
            self.assertIs(events[idx], expected)

    def test_sort_by_name_and_stability(self):
        """Sorting by name and then by cpu_time_total preserves relative order of ties."""
        e1, e2, e3 = _make_event(1, "b", 0, 10), _make_event(2, "a", 0, 10), _make_event(3, "c", 0, 5)
        events = EventList([e1, e2, e3])
        self.assertIsNone(events.sort(key=lambda e: e.name))
        self.assertIs(events[0], e2)
        self.assertIs(events[1], e1)
        self.assertIs(events[2], e3)

        events.sort(key=lambda e: e.cpu_time_total)
        self.assertIs(events[0], e3)
        self.assertIs(events[1], e2)
        self.assertIs(events[2], e1)

    def test_sort_empty_and_single(self):
        """Sorting an empty or single-element EventList succeeds without key."""
        empty = EventList()
        self.assertIsNone(empty.sort())
        self.assertEqual(len(empty), 0)

        e1 = _make_event(1, "only", 0, 10)
        single = EventList([e1])
        self.assertIsNone(single.sort())
        self.assertIs(single[0], e1)

    def test_unorderable_default_key_raises(self):
        """Sorting multiple FunctionEvents without a key raises TypeError."""
        events = EventList([_make_event(1, "a", 0, 10), _make_event(2, "b", 0, 20)])
        with self.assertRaises(TypeError):
            events.sort()

    def test_positional_key_raises(self):
        """Passing key as a positional argument raises TypeError."""
        events = EventList([_make_event(1, "a", 0, 10), _make_event(2, "b", 0, 20)])
        with self.assertRaises(TypeError):
            events.sort(lambda e: e.name)


if __name__ == "__main__":
    run_tests()
