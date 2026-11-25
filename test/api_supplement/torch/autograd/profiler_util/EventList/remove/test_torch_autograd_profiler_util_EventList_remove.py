# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.profiler_util.EventList.remove 接口功能正确性
API 名称：torch.autograd.profiler_util.EventList.remove
API 签名：EventList.remove(self, value) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空列表删除报错与非空列表删除 | 已覆盖（remove_empty_raises、remove_existing） |
| 枚举选项 | 无枚举参数 | 未覆盖（接口无枚举选项） |
| 参数类型 | 合法 FunctionEvent 实例与缺参报错 | 已覆盖（remove_existing、missing_arg_raises） |
| 传参与不传参 | 必选 value 传入与不传参报错 | 已覆盖（remove_existing、missing_arg_raises） |
| 等价类/边界值 | 仅删首次匹配、单元素清空、不存在报错 | 已覆盖（remove_first_occurrence_only、remove_single_element、remove_absent_raises） |
| 正常传参场景 | 删除存在元素、重复元素首项与唯一元素 | 已覆盖（remove_existing、remove_first_occurrence_only、remove_single_element） |
| 异常传参场景 | 元素不存在/空列表 ValueError、缺参 TypeError | 已覆盖（remove_absent_raises、remove_empty_raises、missing_arg_raises） |
| 混合设备类型 | 不涉及 Tensor 输入 | 未覆盖（纯 Python 容器方法，无 Tensor 输入） |

未覆盖项及原因：
- 枚举选项：接口仅接收单个待删除对象 value，无枚举参数。
- 混合设备类型：接口为 list 子类方法，不接收 Tensor 参数。

注意：本接口为 EventList 容器按值原地删除方法，无浮点 Tensor 输出，
     按 API 语义使用返回值 None、剩余元素引用及异常断言验证。
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
class TestEventListRemove(TestCase):
    """Test cases for torch.autograd.profiler_util.EventList.remove."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_remove_existing(self):
        """Removing an existing element deletes it in place and returns None."""
        e1 = _make_event(1, "op_1")
        e2 = _make_event(2, "op_2")
        e3 = _make_event(3, "op_3")
        events = EventList([e1, e2, e3])

        ret = events.remove(e2)
        self.assertIsNone(ret)
        self.assertEqual(len(events), 2)
        self.assertIs(events[0], e1)
        self.assertIs(events[1], e3)

    def test_remove_first_occurrence_only(self):
        """Removing a duplicated element deletes only its first occurrence."""
        e1 = _make_event(1, "op_dup")
        e2 = _make_event(2, "op_mid")
        events = EventList([e1, e2, e1])

        ret = events.remove(e1)
        self.assertIsNone(ret)
        self.assertEqual(len(events), 2)
        self.assertIs(events[0], e2)
        self.assertIs(events[1], e1)

    def test_remove_single_element(self):
        """Removing the sole element leaves EventList empty."""
        e1 = _make_event(1, "op_only")
        events = EventList([e1])
        self.assertIsNone(events.remove(e1))
        self.assertEqual(len(events), 0)

    def test_remove_absent_raises(self):
        """Removing an absent element raises ValueError without modifying EventList."""
        e1 = _make_event(1, "op_1")
        e_other = _make_event(2, "op_other")
        events = EventList([e1])

        with self.assertRaises(ValueError):
            events.remove(e_other)
        self.assertEqual(len(events), 1)
        self.assertIs(events[0], e1)

    def test_remove_empty_raises(self):
        """Removing from an empty EventList raises ValueError."""
        events = EventList()
        with self.assertRaises(ValueError):
            events.remove(_make_event(1, "op_1"))

    def test_missing_arg_raises(self):
        """Calling remove() without value raises TypeError."""
        events = EventList([_make_event(1, "op_1")])
        with self.assertRaises(TypeError):
            events.remove()
        self.assertEqual(len(events), 1)


if __name__ == "__main__":
    run_tests()
