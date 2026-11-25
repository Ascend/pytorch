# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.profiler_util.EventList.key_averages 接口功能正确性
API 名称：torch.autograd.profiler_util.EventList.key_averages
API 签名：EventList.key_averages(self, group_by_input_shapes=False, group_by_stack_n=0, ...)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空事件列表与多事件分组聚合 | 已覆盖（empty_and_unbuilt_tree、default_grouping） |
| 枚举选项 | 分组开关与 Python 函数过滤选项 | 已覆盖（group_by_options、include_python_functions） |
| 参数类型 | 布尔开关与整型栈深度参数 | 已覆盖（group_by_options、include_python_functions） |
| 传参与不传参 | 全默认参数与显式分组参数 | 已覆盖（default_grouping、group_by_options、include_python_functions） |
| 等价类/边界值 | 同名合并、按 shape/stack/overload 拆分 | 已覆盖（default_grouping、group_by_options） |
| 正常传参场景 | 默认分组、细粒度分组及包含 Python 函数 | 已覆盖（default_grouping、group_by_options、include_python_functions） |
| 异常传参场景 | 未构建事件树调用触发 AssertionError | 已覆盖（empty_and_unbuilt_tree） |
| 混合设备类型 | 不涉及多 Tensor 混合设备输入 | 未覆盖（聚合 FunctionEvent 元数据，无 Tensor 输入） |

未覆盖项及原因：
- 混合设备类型：接口仅对 EventList 内事件统计项做分组聚合，不接收 Tensor 参数。

注意：本接口返回聚合后的 FunctionEventAvg 列表，无浮点 Tensor 输出，
     按 API 语义使用返回类型、分组数量、累计耗时及异常断言验证。
"""
import os
import unittest

import torch
from torch.autograd.profiler_util import EventList, FunctionEvent, FunctionEventAvg

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


def _make_event(eid, name, start, end, shapes=None, stack=None, overload="", is_py=False):
    return FunctionEvent(id=eid, name=name, thread=1, start_us=start, end_us=end,
                         input_shapes=shapes or [], stack=stack or [],
                         overload_name=overload, is_python_function=is_py)


def _build_events(evts):
    events = EventList(evts)
    events._build_tree()
    return events


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestEventListKeyAverages(TestCase):
    """Test cases for torch.autograd.profiler_util.EventList.key_averages."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_empty_and_unbuilt_tree(self):
        """Empty built EventList returns empty EventList; unbuilt tree raises AssertionError."""
        avgs = _build_events([]).key_averages()
        self.assertIsInstance(avgs, EventList)
        self.assertEqual(len(avgs), 0)
        with self.assertRaises(AssertionError):
            EventList([_make_event(1, "aten::add", 0, 10)]).key_averages()

    def test_default_grouping(self):
        """Default key_averages() aggregates events by name into FunctionEventAvg entries."""
        events = _build_events([
            _make_event(1, "aten::add", 0, 10),
            _make_event(2, "aten::add", 20, 50),
            _make_event(3, "aten::mul", 60, 75),
        ])
        avgs = events.key_averages()
        self.assertIsInstance(avgs, EventList)
        self.assertEqual(len(avgs), 2)
        self.assertTrue(all(isinstance(item, FunctionEventAvg) for item in avgs))
        by_key = {item.key: item for item in avgs}
        self.assertEqual(by_key["aten::add"].count, 2)
        self.assertAlmostEqual(by_key["aten::add"].cpu_time_total, 40.0, places=6)
        self.assertEqual(by_key["aten::mul"].count, 1)
        self.assertAlmostEqual(by_key["aten::mul"].cpu_time_total, 15.0, places=6)

    def test_group_by_options(self):
        """Grouping by input shapes, stack depth, or overload name splits distinct keys."""
        events = _build_events([
            _make_event(1, "aten::add", 0, 10, shapes=[[2, 2]], stack=["a.py:1"], overload="Tensor"),
            _make_event(2, "aten::add", 20, 45, shapes=[[4, 4]], stack=["b.py:2"], overload="out"),
            _make_event(3, "aten::add", 50, 60, shapes=[[2, 2]], stack=["a.py:1"], overload="Tensor"),
        ])
        self.assertEqual(len(events.key_averages()), 1)
        self.assertEqual(len(events.key_averages(group_by_input_shapes=True)), 2)
        self.assertEqual(len(events.key_averages(group_by_stack_n=1)), 2)
        self.assertEqual(len(events.key_averages(group_by_overload_name=True)), 2)

    def test_include_python_functions(self):
        """include_python_functions controls whether Python function events are included."""
        events = _build_events([_make_event(1, "aten::add", 0, 10), _make_event(2, "my_py_fn", 20, 50, is_py=True)])
        excluded_keys = {item.key for item in events.key_averages(include_python_functions=False)}
        self.assertIn("aten::add", excluded_keys)
        self.assertNotIn("my_py_fn", excluded_keys)
        included_keys = {item.key for item in events.key_averages(include_python_functions=True)}
        self.assertIn("aten::add", included_keys)
        self.assertIn("my_py_fn", included_keys)


if __name__ == "__main__":
    run_tests()
