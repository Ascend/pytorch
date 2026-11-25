# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.profiler_util.EventList.reverse 接口功能正确性
API 名称：torch.autograd.profiler_util.EventList.reverse
API 签名：EventList.reverse(self) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空列表、单元素、多元素原地反转 | 已覆盖（reverse_empty、reverse_single、reverse_multiple_in_place） |
| 枚举选项 | 无枚举参数 | 未覆盖（无参方法，无枚举选项） |
| 参数类型 | 无参方法传入多余参数报错 | 已覆盖（unexpected_arg_raises） |
| 传参与不传参 | 正常无参调用与传多余参数 | 已覆盖（reverse_multiple_in_place、unexpected_arg_raises） |
| 等价类/边界值 | 连续两次反转还原顺序、空/单元素边界 | 已覆盖（double_reverse_restores_order、reverse_empty、reverse_single） |
| 正常传参场景 | 原地反转保留对象引用与顺序还原 | 已覆盖（reverse_multiple_in_place、double_reverse_restores_order、reverse_empty、reverse_single） |
| 异常传参场景 | 传入多余位置参数触发 TypeError | 已覆盖（unexpected_arg_raises） |
| 混合设备类型 | 不涉及 Tensor 输入 | 未覆盖（纯 Python 容器方法，无 Tensor 输入） |

未覆盖项及原因：
- 枚举选项：接口不接受任何参数，无枚举选项。
- 混合设备类型：接口为 list 子类方法，不接收 Tensor 参数。

注意：本接口为 EventList 容器原地反转方法，无浮点 Tensor 输出，
     按 API 语义使用返回值 None、引用顺序及异常断言验证。
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


def _make_event(event_id, name):
    return FunctionEvent(id=event_id, name=name, thread=1, start_us=0, end_us=10)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestEventListReverse(TestCase):
    """Test cases for torch.autograd.profiler_util.EventList.reverse."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_reverse_multiple_in_place(self):
        """Reversing a multi-element EventList mutates it in place and returns None."""
        e1 = _make_event(1, "op_1")
        e2 = _make_event(2, "op_2")
        e3 = _make_event(3, "op_3")
        events = EventList([e1, e2, e3])
        alias = events

        ret = events.reverse()
        self.assertIsNone(ret)
        self.assertIs(events, alias)
        self.assertEqual(len(events), 3)
        self.assertIs(events[0], e3)
        self.assertIs(events[1], e2)
        self.assertIs(events[2], e1)

    def test_double_reverse_restores_order(self):
        """Calling reverse() twice restores the original element order."""
        e1 = _make_event(1, "op_1")
        e2 = _make_event(2, "op_2")
        e3 = _make_event(3, "op_3")
        events = EventList([e1, e2, e3])

        events.reverse()
        events.reverse()
        self.assertIs(events[0], e1)
        self.assertIs(events[1], e2)
        self.assertIs(events[2], e3)

    def test_reverse_empty(self):
        """Reversing an empty EventList is a no-op returning None."""
        events = EventList()
        ret = events.reverse()
        self.assertIsNone(ret)
        self.assertEqual(len(events), 0)

    def test_reverse_single(self):
        """Reversing a single-element EventList keeps the element unchanged."""
        e1 = _make_event(1, "op_only")
        events = EventList([e1])
        ret = events.reverse()
        self.assertIsNone(ret)
        self.assertEqual(len(events), 1)
        self.assertIs(events[0], e1)

    def test_unexpected_arg_raises(self):
        """Passing an unexpected argument to reverse() raises TypeError."""
        e1 = _make_event(1, "op_1")
        e2 = _make_event(2, "op_2")
        events = EventList([e1, e2])
        with self.assertRaises(TypeError):
            events.reverse(True)
        self.assertIs(events[0], e1)
        self.assertIs(events[1], e2)


if __name__ == "__main__":
    run_tests()
