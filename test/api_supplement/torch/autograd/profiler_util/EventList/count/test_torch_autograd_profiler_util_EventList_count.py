# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.profiler_util.EventList.count 接口功能正确性
API 名称：torch.autograd.profiler_util.EventList.count
API 签名：EventList.count(self, value) -> int

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空列表与非空列表计数 | 已覆盖（count_empty、count_unique_and_multiple） |
| 枚举选项 | 无枚举参数 | 未覆盖（接口无枚举选项） |
| 参数类型 | FunctionEvent、同字段不同实例、str、None | 已覆盖（count_unique_and_multiple、count_absent_and_other_type） |
| 传参与不传参 | 正常单参、缺参、多参 | 已覆盖（count_unique_and_multiple、missing_arg_raises、extra_arg_raises） |
| 等价类/边界值 | 出现 0/1/多次及身份相等判定 | 已覆盖（count_empty、count_unique_and_multiple、count_absent_and_other_type） |
| 正常传参场景 | 唯一、重复与缺失元素计数且不修改列表 | 已覆盖（count_unique_and_multiple、count_absent_and_other_type） |
| 异常传参场景 | 缺参与多余位置参数触发 TypeError | 已覆盖（missing_arg_raises、extra_arg_raises） |
| 混合设备类型 | 不涉及 Tensor 输入 | 未覆盖（纯 Python 容器方法，无 Tensor 输入） |

未覆盖项及原因：
- 枚举选项：接口仅接收单个待计数对象 value，无枚举参数。
- 混合设备类型：接口为 list 子类方法，不接收 Tensor 参数。

注意：本接口为 EventList 容器计数方法，无浮点 Tensor 输出，
     按 API 语义使用整型返回值、无副作用及异常断言验证。
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
        id=event_id,
        name=name,
        trace_name=name,
        thread=1,
        start_us=start_us,
        end_us=end_us,
    )


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestEventListCount(TestCase):
    """Test cases for torch.autograd.profiler_util.EventList.count."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_count_empty(self):
        """Counting any value in an empty EventList returns 0."""
        el = EventList()
        e1 = _make_event(1, "aten::add")
        self.assertEqual(el.count(e1), 0)
        self.assertEqual(el.count(None), 0)

    def test_count_unique_and_multiple(self):
        """count() returns 1 for unique reference and exact count for duplicate references."""
        e1 = _make_event(1, "aten::add", 0, 10)
        e2 = _make_event(2, "aten::mul", 10, 20)
        el = EventList([e1, e2, e1, e1])

        self.assertEqual(el.count(e1), 3)
        self.assertEqual(el.count(e2), 1)
        # Ensure count() does not mutate the EventList.
        self.assertEqual(len(el), 4)

    def test_count_absent_and_other_type(self):
        """Absent FunctionEvent, distinct same-field instance, and non-event values return 0."""
        e1 = _make_event(1, "aten::add", 0, 10)
        e1_clone = _make_event(1, "aten::add", 0, 10)
        e_absent = _make_event(99, "aten::sub", 50, 60)
        el = EventList([e1])

        self.assertEqual(el.count(e_absent), 0)
        self.assertEqual(el.count(e1_clone), 0)
        self.assertEqual(el.count("aten::add"), 0)
        self.assertEqual(el.count(None), 0)

    def test_missing_arg_raises(self):
        """Calling count() without value raises TypeError."""
        el = EventList()
        with self.assertRaises(TypeError):
            el.count()

    def test_extra_arg_raises(self):
        """Calling count() with too many positional arguments raises TypeError."""
        e1 = _make_event(1, "op")
        el = EventList([e1])
        with self.assertRaises(TypeError):
            el.count(e1, 0)


if __name__ == "__main__":
    run_tests()
