# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.profiler_util.EventList.insert 接口功能正确性
API 名称：torch.autograd.profiler_util.EventList.insert
API 签名：EventList.insert(self, index, object) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空列表插入与非空列表指定位置插入 | 已覆盖（insert_into_empty、insert_positions） |
| 枚举选项 | 无枚举参数 | 未覆盖（接口无枚举选项） |
| 参数类型 | 整型索引与字符串/浮点索引报错 | 已覆盖（insert_positions、non_int_index_raises） |
| 传参与不传参 | 完整双参传入与缺参报错 | 已覆盖（insert_into_empty、missing_args_raises） |
| 等价类/边界值 | 首/中/尾位置、负索引与超大/超小裁剪 | 已覆盖（insert_positions、negative_and_clamped_index） |
| 正常传参场景 | 各类合法索引原地插入并保持对象引用 | 已覆盖（insert_into_empty、insert_positions、negative_and_clamped_index） |
| 异常传参场景 | 非整数索引与缺参触发 TypeError | 已覆盖（non_int_index_raises、missing_args_raises） |
| 混合设备类型 | 不涉及 Tensor 输入 | 未覆盖（纯 Python 容器方法，无 Tensor 输入） |

未覆盖项及原因：
- 枚举选项：接口仅接收索引 index 与待插入对象 object，无枚举参数。
- 混合设备类型：接口为 list 子类方法，不接收 Tensor 参数。

注意：本接口为 EventList 容器原地插入方法，无浮点 Tensor 输出，
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


def _make_event(event_id, name):
    return FunctionEvent(id=event_id, name=name, thread=1, start_us=0, end_us=10)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestEventListInsert(TestCase):
    """Test cases for torch.autograd.profiler_util.EventList.insert."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_insert_into_empty(self):
        """Inserting into an empty EventList adds the element at index 0."""
        events = EventList()
        ev = _make_event(1, "op_a")
        self.assertIsNone(events.insert(0, ev))
        self.assertEqual(len(events), 1)
        self.assertIs(events[0], ev)

    def test_insert_positions(self):
        """Inserting at head, middle, and tail places elements at expected positions."""
        e1, e2 = _make_event(1, "op_1"), _make_event(2, "op_2")
        e_head, e_mid, e_tail = _make_event(3, "h"), _make_event(4, "m"), _make_event(5, "t")
        events = EventList([e1, e2])

        self.assertIsNone(events.insert(0, e_head))
        self.assertIsNone(events.insert(2, e_mid))
        self.assertIsNone(events.insert(len(events), e_tail))
        self.assertEqual(len(events), 5)
        for idx, expected in enumerate([e_head, e1, e_mid, e2, e_tail]):
            self.assertIs(events[idx], expected)

    def test_negative_and_clamped_index(self):
        """Negative and out-of-range indices follow Python list.insert clamping rules."""
        e1, e2 = _make_event(1, "op_1"), _make_event(2, "op_2")
        e_neg, e_front, e_back = _make_event(3, "n"), _make_event(4, "f"), _make_event(5, "b")
        events = EventList([e1, e2])

        self.assertIsNone(events.insert(-1, e_neg))
        self.assertIsNone(events.insert(100, e_back))
        self.assertIsNone(events.insert(-100, e_front))
        for idx, expected in enumerate([e_front, e1, e_neg, e2, e_back]):
            self.assertIs(events[idx], expected)

    def test_non_int_index_raises(self):
        """Non-integer indices raise TypeError without modifying EventList."""
        e1, e2 = _make_event(1, "op_1"), _make_event(2, "op_2")
        events = EventList([e1])
        with self.assertRaises(TypeError):
            events.insert("0", e2)
        with self.assertRaises(TypeError):
            events.insert(0.5, e2)
        self.assertEqual(len(events), 1)
        self.assertIs(events[0], e1)

    def test_missing_args_raises(self):
        """Calling insert() with fewer than two arguments raises TypeError."""
        events = EventList()
        with self.assertRaises(TypeError):
            events.insert()
        with self.assertRaises(TypeError):
            events.insert(0)
        self.assertEqual(len(events), 0)


if __name__ == "__main__":
    run_tests()
