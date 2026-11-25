# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.profiler_util.EventList.copy 接口功能正确性
API 名称：torch.autograd.profiler_util.EventList.copy
API 签名：EventList.copy(self) -> list

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空与非空 EventList 浅拷贝 | 已覆盖（copy_empty、copy_non_empty） |
| 枚举选项 | 无枚举参数 | 未覆盖（无参方法，无枚举选项） |
| 参数类型 | 无参方法传入多余参数报错 | 已覆盖（unexpected_arg_raises） |
| 传参与不传参 | 正常无参调用与传多余参数 | 已覆盖（copy_non_empty、unexpected_arg_raises） |
| 等价类/边界值 | 浅拷贝元素引用共享与增删隔离 | 已覆盖（shallow_element_identity、mutation_isolation） |
| 正常传参场景 | 拷贝非空/空列表并验证浅拷贝语义 | 已覆盖（copy_non_empty、shallow_element_identity、mutation_isolation、copy_empty） |
| 异常传参场景 | 传入多余位置参数触发 TypeError | 已覆盖（unexpected_arg_raises） |
| 混合设备类型 | 不涉及 Tensor 输入 | 未覆盖（纯 Python 容器方法，无 Tensor 输入） |

未覆盖项及原因：
- 枚举选项：接口不接受任何参数，无枚举选项。
- 混合设备类型：接口为 list 子类方法，不接收 Tensor 参数。

注意：本接口为 EventList 容器浅拷贝方法，无浮点 Tensor 输出，
     按 API 语义使用类型、引用标识、变更隔离及异常断言验证。
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
class TestEventListCopy(TestCase):
    """Test cases for torch.autograd.profiler_util.EventList.copy."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_copy_non_empty(self):
        """Copying a non-empty EventList returns a new equal list instance."""
        e1 = _make_event(1, "aten::add", 0, 10)
        e2 = _make_event(2, "aten::mul", 10, 20)
        el = EventList([e1, e2])
        copied = el.copy()
        self.assertIsInstance(copied, list)
        self.assertIsNot(copied, el)
        self.assertEqual(copied, el)
        self.assertEqual(len(copied), 2)

    def test_shallow_element_identity(self):
        """copy() performs a shallow copy preserving element object identity."""
        e1 = _make_event(1, "op_a", 0, 15)
        e2 = _make_event(2, "op_b", 15, 35)
        el = EventList([e1, e2])
        copied = el.copy()
        self.assertIs(copied[0], e1)
        self.assertIs(copied[1], e2)
        self.assertIs(copied[0], el[0])
        self.assertIs(copied[1], el[1])

    def test_mutation_isolation(self):
        """Mutating the copied list or original EventList does not affect the other."""
        e1 = _make_event(1, "op_1", 0, 10)
        e2 = _make_event(2, "op_2", 10, 20)
        e3 = _make_event(3, "op_3", 20, 30)
        el = EventList([e1, e2])
        copied = el.copy()
        copied.append(e3)
        self.assertEqual(len(copied), 3)
        self.assertEqual(len(el), 2)
        self.assertNotIn(e3, el)
        el.pop()
        self.assertEqual(len(el), 1)
        self.assertEqual(len(copied), 3)
        self.assertIs(copied[1], e2)

    def test_copy_empty(self):
        """Copying an empty EventList returns a distinct empty list."""
        el = EventList()
        copied = el.copy()
        self.assertIsInstance(copied, list)
        self.assertIsNot(copied, el)
        self.assertEqual(copied, [])
        self.assertEqual(len(copied), 0)

    def test_unexpected_arg_raises(self):
        """Passing an unexpected argument to copy() raises TypeError."""
        el = EventList([_make_event(1, "op")])
        with self.assertRaises(TypeError):
            el.copy(1)


if __name__ == "__main__":
    run_tests()
