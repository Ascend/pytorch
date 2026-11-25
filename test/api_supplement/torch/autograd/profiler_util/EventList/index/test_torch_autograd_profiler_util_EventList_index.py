# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.profiler_util.EventList.index 接口功能正确性
API 名称：torch.autograd.profiler_util.EventList.index
API 签名：EventList.index(self, value, start=0, stop=sys.maxsize) -> int

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空列表查找报错与非空列表查找 | 已覆盖（index_empty_raises、index_first_occurrence） |
| 枚举选项 | 无枚举参数 | 未覆盖（接口无枚举选项） |
| 参数类型 | 合法整数边界与非整数 start 报错 | 已覆盖（index_with_bounds、non_int_start_raises） |
| 传参与不传参 | 仅传 value、传 start/stop 与缺参 | 已覆盖（index_first_occurrence、index_with_bounds、missing_arg_raises） |
| 等价类/边界值 | 首次匹配、正负边界、越界裁剪与区间外 | 已覆盖（index_first_occurrence、index_with_bounds、not_found_and_outside_bounds_raises） |
| 正常传参场景 | 全量与区间内查找返回首次索引 | 已覆盖（index_first_occurrence、index_with_bounds） |
| 异常传参场景 | ValueError / TypeError | 已覆盖（not_found_and_outside_bounds_raises、index_empty_raises、missing_arg_raises 等） |
| 混合设备类型 | 不涉及 Tensor 输入 | 未覆盖（纯 Python 容器方法，无 Tensor 输入） |

未覆盖项及原因：
- 枚举选项：接口仅接收 value 与起止切片索引 start/stop，无枚举参数。
- 混合设备类型：接口为 list 子类方法，不接收 Tensor 参数。

注意：本接口为 EventList 容器索引查找方法，无浮点 Tensor 输出，
     按 API 语义使用返回索引值、无副作用及异常断言验证。
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
    """Create a minimal FunctionEvent for EventList testing."""
    return FunctionEvent(
        id=event_id, name=name, trace_name=name,
        thread=1, start_us=start_us, end_us=end_us,
    )


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestEventListIndex(TestCase):
    """Test cases for torch.autograd.profiler_util.EventList.index."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_index_first_occurrence(self):
        """index() returns the index of the first matching element without mutating list."""
        e1, e2, e3 = _make_event(1, "op1"), _make_event(2, "op2"), _make_event(3, "op3")
        el = EventList([e1, e2, e3, e2])
        self.assertEqual(el.index(e1), 0)
        self.assertEqual(el.index(e2), 1)
        self.assertEqual(el.index(e3), 2)
        self.assertEqual(len(el), 4)

    def test_index_with_bounds(self):
        """Positive, negative, and out-of-range start/stop bounds follow slice semantics."""
        e1, e2, e3 = _make_event(1, "op1"), _make_event(2, "op2"), _make_event(3, "op3")
        el = EventList([e1, e2, e3, e2, e1])
        self.assertEqual(el.index(e2, 2), 3)
        self.assertEqual(el.index(e1, 1, 5), 4)
        self.assertEqual(el.index(e2, -3, -1), 3)
        self.assertEqual(el.index(e1, -100, 100), 0)

    def test_not_found_and_outside_bounds_raises(self):
        """Absent values or values outside [start, stop) raise ValueError."""
        e1, e2, e_absent = _make_event(1, "op1"), _make_event(2, "op2"), _make_event(9, "none")
        el = EventList([e1, e2])
        with self.assertRaises(ValueError):
            el.index(e_absent)
        with self.assertRaises(ValueError):
            el.index(e2, 0, 1)

    def test_index_empty_raises(self):
        """Searching in an empty EventList raises ValueError."""
        with self.assertRaises(ValueError):
            EventList().index(_make_event(1, "op1"))

    def test_missing_arg_raises(self):
        """Calling index() without value raises TypeError."""
        with self.assertRaises(TypeError):
            EventList().index()

    def test_non_int_start_raises(self):
        """Passing a non-integer start bound raises TypeError."""
        e1 = _make_event(1, "op1")
        with self.assertRaises(TypeError):
            EventList([e1]).index(e1, "0")


if __name__ == "__main__":
    run_tests()
