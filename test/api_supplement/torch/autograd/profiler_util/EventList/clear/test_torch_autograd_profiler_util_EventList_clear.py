# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.profiler_util.EventList.clear 接口功能正确性
API 名称：torch.autograd.profiler_util.EventList.clear
API 签名：EventList.clear(self) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 非空、单元素、空列表重复清空 | 已覆盖（clear_non_empty、clear_single_and_empty_idempotent） |
| 枚举选项 | 无枚举参数 | 未覆盖（无参方法，无枚举选项） |
| 参数类型 | 无参方法传入多余参数报错 | 已覆盖（unexpected_arg_raises） |
| 传参与不传参 | 正常无参调用与传多余参数 | 已覆盖（clear_non_empty、unexpected_arg_raises） |
| 等价类/边界值 | 空列表幂等、清空后复用、元数据保留 | 已覆盖（clear_single_and_empty_idempotent、preserves_metadata、clear_then_reuse） |
| 正常传参场景 | 原地清空列表并保留实例与元数据 | 已覆盖（clear_non_empty、preserves_metadata、clear_then_reuse） |
| 异常传参场景 | 传入多余位置参数触发 TypeError | 已覆盖（unexpected_arg_raises） |
| 混合设备类型 | 不涉及 Tensor 输入 | 未覆盖（纯 Python 容器方法，无 Tensor 输入） |

未覆盖项及原因：
- 枚举选项：接口不接受任何参数，无枚举选项。
- 混合设备类型：接口为 list 子类方法，不接收 Tensor 参数。

注意：本接口为 EventList 容器原地清空方法，无浮点 Tensor 输出，
     按 API 语义使用长度、引用一致性、元数据属性及异常断言验证。
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


def _make_event(event_id, name, start_us=0, end_us=10):
    return FunctionEvent(id=event_id, name=name, trace_name=name, thread=1, start_us=start_us, end_us=end_us)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestEventListClear(TestCase):
    """Test cases for torch.autograd.profiler_util.EventList.clear."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_clear_non_empty(self):
        """Clearing a non-empty EventList empties it in place and returns None."""
        e1, e2 = _make_event(1, "aten::add", 0, 10), _make_event(2, "aten::mul", 10, 25)
        el = EventList([e1, e2])
        alias = el
        self.assertIsNone(el.clear())
        self.assertIs(el, alias)
        self.assertEqual(len(el), 0)
        self.assertEqual(len(alias), 0)
        self.assertEqual(el, [])
        self.assertNotIn(e1, el)
        self.assertNotIn(e2, el)

    def test_clear_single_and_empty_idempotent(self):
        """Clearing a single-element EventList and clearing again is idempotent."""
        el = EventList([_make_event(1, "solo", 0, 5)])
        self.assertEqual(len(el), 1)
        self.assertIsNone(el.clear())
        self.assertEqual(len(el), 0)
        self.assertIsNone(el.clear())
        self.assertEqual(len(el), 0)

    def test_preserves_metadata(self):
        """Clearing events preserves EventList configuration attributes."""
        el = EventList([_make_event(1, "aten::mm", 0, 50)], use_device=self.device_name,
                       profile_memory=True, with_flops=True)
        el.clear()
        self.assertEqual(len(el), 0)
        self.assertEqual(el._use_device, self.device_name)
        self.assertTrue(el._profile_memory)
        self.assertTrue(el._with_flops)
        self.assertFalse(el._tree_built)

    def test_clear_then_reuse(self):
        """An EventList can be appended to and reused normally after clear()."""
        e1, e2 = _make_event(1, "old_op", 0, 10), _make_event(2, "new_op", 20, 40)
        el = EventList([e1])
        el.clear()
        el.append(e2)
        self.assertEqual(len(el), 1)
        self.assertIs(el[0], e2)

    def test_unexpected_arg_raises(self):
        """Passing an unexpected argument to clear() raises TypeError without mutating list."""
        e1 = _make_event(1, "keep_me", 0, 10)
        el = EventList([e1])
        with self.assertRaises(TypeError):
            el.clear(None)
        self.assertEqual(len(el), 1)
        self.assertIs(el[0], e1)


if __name__ == "__main__":
    run_tests()
