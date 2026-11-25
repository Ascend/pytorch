# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.profiler_util.EventList 接口功能正确性
API 名称：torch.autograd.profiler_util.EventList
API 签名：EventList(iterable=(), use_device=None, profile_memory=False, with_flops=False)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 无参空构造与非空可迭代对象构造 | 已覆盖（empty_init、init_from_iterable） |
| 枚举选项 | use_device、profile_memory、with_flops | 已覆盖（empty_init、custom_flags） |
| 参数类型 | list/tuple/generator 与非可迭代类型报错 | 已覆盖（init_from_iterable、non_iterable_raises） |
| 传参与不传参 | 全默认参数、自定义标志与多余位置参数 | 已覆盖（empty_init、custom_flags、too_many_args_raises） |
| 等价类/边界值 | 索引、负索引、切片、迭代与耗时统计 | 已覆盖（indexing_and_iteration） |
| 正常传参场景 | 默认构造、自定义配置、迭代与切片访问 | 已覆盖（empty_init、custom_flags、init_from_iterable、indexing_and_iteration） |
| 异常传参场景 | 非可迭代首参、多余位置参数 TypeError | 已覆盖（non_iterable_raises、too_many_args_raises） |
| 混合设备类型 | 不涉及 Tensor 输入 | 未覆盖（纯 Python 容器类构造，无 Tensor 输入） |

未覆盖项及原因：
- 混合设备类型：接口为 list 子类容器构造器，不接收 Tensor 参数。

注意：本接口为 EventList 容器类构造与基础序列行为，无浮点 Tensor 输出，
     按 API 语义使用类型、属性标志、元素引用及异常断言验证。
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
class TestEventList(TestCase):
    """Test cases for torch.autograd.profiler_util.EventList."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_empty_init(self):
        """Default EventList() initializes an empty list with default metadata flags."""
        el = EventList()
        self.assertIsInstance(el, list)
        self.assertEqual(len(el), 0)
        self.assertIsNone(el._use_device)
        self.assertFalse(el._profile_memory)
        self.assertFalse(el._with_flops)
        self.assertFalse(el._tree_built)

    def test_custom_flags(self):
        """EventList stores custom use_device, profile_memory, and with_flops flags."""
        e1 = _make_event(1, "aten::mm", 0, 20)
        el = EventList([e1], use_device=self.device_name, profile_memory=True, with_flops=True)
        self.assertEqual(el._use_device, self.device_name)
        self.assertTrue(el._profile_memory)
        self.assertTrue(el._with_flops)
        self.assertFalse(el._tree_built)
        self.assertIs(el[0], e1)

    def test_init_from_iterable(self):
        """EventList initializes cleanly from lists, tuples, and generators."""
        e1, e2 = _make_event(1, "op1", 0, 10), _make_event(2, "op2", 10, 30)
        for el in (EventList([e1, e2]), EventList((e1, e2)), EventList(ev for ev in (e1, e2))):
            self.assertEqual(len(el), 2)
            self.assertIs(el[0], e1)
            self.assertIs(el[1], e2)

    def test_indexing_and_iteration(self):
        """EventList supports indexing, slicing, iteration, and self_cpu_time_total."""
        events = [_make_event(i, f"op_{i}", i * 10, (i + 1) * 10) for i in range(3)]
        el = EventList(events)
        self.assertIs(el[0], events[0])
        self.assertIs(el[-1], events[-1])
        self.assertEqual(el[1:3], events[1:3])
        self.assertEqual(list(el), events)
        self.assertEqual(el.self_cpu_time_total, 30)

    def test_non_iterable_raises(self):
        """Constructing EventList from a non-iterable raises TypeError."""
        with self.assertRaises(TypeError):
            EventList(12345)

    def test_too_many_args_raises(self):
        """Passing extra positional arguments to EventList raises TypeError."""
        e1 = _make_event(1, "op1")
        with self.assertRaises(TypeError):
            EventList([e1], [e1])


if __name__ == "__main__":
    run_tests()
