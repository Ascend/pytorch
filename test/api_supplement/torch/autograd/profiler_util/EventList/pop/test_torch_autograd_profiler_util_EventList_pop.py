# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.profiler_util.EventList.pop 接口功能正确性
API 名称：torch.autograd.profiler_util.EventList.pop
API 签名：EventList.pop(self, index=-1)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空列表弹出报错与非空列表弹出 | 已覆盖（invalid_pop_raises、pop_default_last） |
| 枚举选项 | 无枚举参数 | 未覆盖（接口无枚举选项） |
| 参数类型 | 合法整型索引与字符串/浮点索引报错 | 已覆盖（pop_explicit_index、invalid_pop_raises） |
| 传参与不传参 | 省略 index 默认弹末尾与显式传 index | 已覆盖（pop_default_last、pop_explicit_index、pop_negative_index） |
| 等价类/边界值 | 首/中/负索引、正负越界索引 | 已覆盖（pop_explicit_index、pop_negative_index、invalid_pop_raises） |
| 正常传参场景 | 默认末尾、正索引与负索引原地弹出 | 已覆盖（pop_default_last、pop_explicit_index、pop_negative_index） |
| 异常传参场景 | 空列表/越界 IndexError、非整型 TypeError | 已覆盖（invalid_pop_raises） |
| 混合设备类型 | 不涉及 Tensor 输入 | 未覆盖（纯 Python 容器方法，无 Tensor 输入） |

未覆盖项及原因：
- 枚举选项：接口仅接收可选整数索引 index，无枚举参数。
- 混合设备类型：接口为 list 子类方法，不接收 Tensor 参数。

注意：本接口为 EventList 容器弹出方法，无浮点 Tensor 输出，
     按 API 语义使用弹出元素引用、剩余元素状态及异常断言验证。
"""
import os
import unittest

import torch
from torch.autograd.profiler_util import EventList, FunctionEvent

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


def _make_event(event_id, name):
    return FunctionEvent(id=event_id, name=name, thread=1, start_us=0, end_us=10)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestEventListPop(TestCase):
    """Test cases for torch.autograd.profiler_util.EventList.pop."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_pop_default_last(self):
        """Calling pop() without index removes and returns the last element."""
        e1, e2, e3 = (_make_event(i, f"op_{i}") for i in (1, 2, 3))
        events = EventList([e1, e2, e3])
        self.assertIs(events.pop(), e3)
        self.assertEqual(len(events), 2)
        self.assertIs(events[0], e1)
        self.assertIs(events[1], e2)

    def test_pop_explicit_index(self):
        """Calling pop() with non-negative indices removes and returns target elements."""
        e1, e2, e3, e4 = (_make_event(i, f"op_{i}") for i in (1, 2, 3, 4))
        events = EventList([e1, e2, e3, e4])
        self.assertIs(events.pop(0), e1)
        self.assertEqual(len(events), 3)
        self.assertIs(events.pop(1), e3)
        self.assertEqual(len(events), 2)
        self.assertIs(events[0], e2)
        self.assertIs(events[1], e4)

    def test_pop_negative_index(self):
        """Calling pop() with a negative index counts from the end of EventList."""
        e1, e2, e3 = (_make_event(i, f"op_{i}") for i in (1, 2, 3))
        events = EventList([e1, e2, e3])
        self.assertIs(events.pop(-2), e2)
        self.assertEqual(len(events), 2)
        self.assertIs(events[0], e1)
        self.assertIs(events[1], e3)

    def test_invalid_pop_raises(self):
        """Popping from empty list, out-of-range index, or non-int index raises errors."""
        with self.assertRaises(IndexError):
            EventList().pop()
        e1 = _make_event(1, "op_1")
        events = EventList([e1])
        for bad_idx in (1, -2):
            with self.assertRaises(IndexError):
                events.pop(bad_idx)
        for bad_type in ("0", 0.5):
            with self.assertRaises(TypeError):
                events.pop(bad_type)
        self.assertEqual(len(events), 1)
        self.assertIs(events[0], e1)


if __name__ == "__main__":
    run_tests()
