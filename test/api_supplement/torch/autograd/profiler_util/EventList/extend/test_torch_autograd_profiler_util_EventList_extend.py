# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.profiler_util.EventList.extend 接口功能正确性
API 名称：torch.autograd.profiler_util.EventList.extend
API 签名：EventList.extend(self, iterable) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空列表扩展与向空/非空列表追加空序列 | 已覆盖（extend_various_iterables、extend_with_empty） |
| 枚举选项 | 无枚举参数 | 未覆盖（接口无枚举选项） |
| 参数类型 | list、tuple、generator、EventList 与非可迭代对象 | 已覆盖（extend_with_list、extend_various_iterables、non_iterable_raises） |
| 传参与不传参 | 必选 iterable 传入与不传参报错 | 已覆盖（extend_with_list、missing_arg_raises） |
| 等价类/边界值 | 原地扩展保留引用、源 EventList 不被修改 | 已覆盖（extend_with_list、extend_various_iterables、extend_with_empty） |
| 正常传参场景 | 使用各类可迭代对象原地追加元素 | 已覆盖（extend_with_list、extend_various_iterables、extend_with_empty） |
| 异常传参场景 | 传非可迭代对象或缺参触发 TypeError | 已覆盖（non_iterable_raises、missing_arg_raises） |
| 混合设备类型 | 不涉及 Tensor 输入 | 未覆盖（纯 Python 容器方法，无 Tensor 输入） |

未覆盖项及原因：
- 枚举选项：接口仅接收单个可迭代对象 iterable，无枚举选项。
- 混合设备类型：接口为 list 子类方法，不接收 Tensor 参数。

注意：本接口为 EventList 容器原地扩展方法，无浮点 Tensor 输出，
     按 API 语义使用长度、引用顺序、无副作用及异常断言验证。
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
    return FunctionEvent(
        id=event_id, name=name, trace_name=name,
        thread=1, start_us=start_us, end_us=end_us,
    )


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestEventListExtend(TestCase):
    """Test cases for torch.autograd.profiler_util.EventList.extend."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_extend_with_list(self):
        """Extending with a list appends elements in place and returns None."""
        e1, e2, e3 = _make_event(1, "op1"), _make_event(2, "op2"), _make_event(3, "op3")
        el = EventList([e1])
        alias = el
        self.assertIsNone(el.extend([e2, e3]))
        self.assertIs(el, alias)
        self.assertEqual(len(el), 3)
        self.assertIs(el[0], e1)
        self.assertIs(el[1], e2)
        self.assertIs(el[2], e3)

    def test_extend_various_iterables(self):
        """extend() accepts tuples, generators, and other EventList instances."""
        e1, e2 = _make_event(1, "op1"), _make_event(2, "op2")
        e3, e4 = _make_event(3, "op3"), _make_event(4, "op4")
        other = EventList([e4])
        el = EventList()
        self.assertIsNone(el.extend((e1, e2)))
        self.assertIsNone(el.extend(ev for ev in (e3,)))
        self.assertIsNone(el.extend(other))
        self.assertEqual(len(el), 4)
        self.assertIs(el[0], e1)
        self.assertIs(el[2], e3)
        self.assertIs(el[3], e4)
        self.assertEqual(len(other), 1)

    def test_extend_with_empty(self):
        """Extending with an empty list leaves EventList unchanged."""
        e1 = _make_event(1, "only")
        el = EventList([e1])
        self.assertIsNone(el.extend([]))
        self.assertEqual(len(el), 1)
        self.assertIs(el[0], e1)

    def test_non_iterable_raises(self):
        """Passing a non-iterable raises TypeError without mutating EventList."""
        e1 = _make_event(1, "keep")
        el = EventList([e1])
        with self.assertRaises(TypeError):
            el.extend(42)
        self.assertEqual(len(el), 1)
        self.assertIs(el[0], e1)

    def test_missing_arg_raises(self):
        """Calling extend() without arguments raises TypeError."""
        with self.assertRaises(TypeError):
            EventList().extend()


if __name__ == "__main__":
    run_tests()
