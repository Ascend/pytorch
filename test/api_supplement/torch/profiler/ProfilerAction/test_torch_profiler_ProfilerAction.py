# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.profiler.ProfilerAction 枚举接口功能正确性
API 名称：torch.profiler.ProfilerAction
API 签名：class torch.profiler.ProfilerAction(Enum)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------------------------------|--------------------------------------------------|
| 空/非空 | 空名称查找与非空有效枚举成员 | 已覆盖（invalid_name_raises、members_and_types） |
| 枚举选项 | NONE/WARMUP/RECORD/RECORD_AND_SAVE 全成员 | 已覆盖（members_and_types、distinct_values、iteration_all_members） |
| 参数类型 | 字符串名索引与整数/非法值构造 | 已覆盖（name_and_value_roundtrip、invalid_value_raises） |
| 传参与不传参 | 按名称/值访问与枚举遍历 | 已覆盖（name_and_value_roundtrip、iteration_all_members） |
| 等价类/边界值 | 成员值唯一性、相等性与哈希集合/字典键 | 已覆盖（distinct_values、equality_and_hashability） |
| 正常传参场景 | 成员类型、遍历及名称/值往返转换 | 已覆盖（members_and_types、name_and_value_roundtrip、equality_and_hashability） |
| 异常传参场景 | 非法名称 KeyError 与非法值 ValueError | 已覆盖（invalid_name_raises、invalid_value_raises） |
| 混合设备类型 | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖，ProfilerAction 为纯 Python 枚举类，不涉及设备迁移 |

未覆盖项及原因：
- 混合设备类型：ProfilerAction 为纯 Python Enum 枚举定义，不接受 Tensor 输入。

注意：对可在 CPU/NPU 两侧执行的非图模式浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
from enum import Enum
import os
import unittest

import torch
from torch.profiler import ProfilerAction

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


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestProfilerAction(TestCase):
    """Test cases for torch.profiler.ProfilerAction."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_members_and_types(self):
        """Standard ProfilerAction members exist and are Enum instances."""
        self.assertTrue(issubclass(ProfilerAction, Enum))
        expected = ("NONE", "WARMUP", "RECORD", "RECORD_AND_SAVE")
        for member_name in expected:
            self.assertTrue(hasattr(ProfilerAction, member_name))
            member = getattr(ProfilerAction, member_name)
            self.assertIsInstance(member, Enum)
            self.assertIsInstance(member, ProfilerAction)
            self.assertEqual(member.name, member_name)

    def test_distinct_values(self):
        """All four standard ProfilerAction members have distinct values."""
        members = [ProfilerAction.NONE, ProfilerAction.WARMUP, ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE]
        values = [m.value for m in members]
        self.assertEqual(len(set(values)), 4)

    def test_name_and_value_roundtrip(self):
        """Members round-trip cleanly by both name indexing and value lookup."""
        for member in ProfilerAction:
            self.assertIs(ProfilerAction[member.name], member)
            self.assertIs(ProfilerAction(member.value), member)
        self.assertIs(ProfilerAction["NONE"], ProfilerAction.NONE)
        self.assertIs(ProfilerAction(ProfilerAction.WARMUP.value), ProfilerAction.WARMUP)

    def test_iteration_all_members(self):
        """Iterating over ProfilerAction yields all expected standard members."""
        iterated = list(ProfilerAction)
        expected = [
            ProfilerAction.NONE,
            ProfilerAction.WARMUP,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
        ]
        for member in expected:
            self.assertIn(member, iterated)
        self.assertGreaterEqual(len(iterated), 4)

    def test_equality_and_hashability(self):
        """ProfilerAction members support equality comparison, sets, and dict keys."""
        self.assertEqual(ProfilerAction.RECORD, ProfilerAction.RECORD)
        self.assertNotEqual(ProfilerAction.NONE, ProfilerAction.RECORD_AND_SAVE)
        action_set = {ProfilerAction.NONE, ProfilerAction.WARMUP, ProfilerAction.NONE}
        self.assertEqual(len(action_set), 2)
        mapping = {ProfilerAction.RECORD: "rec"}
        self.assertEqual(mapping[ProfilerAction.RECORD], "rec")

    def test_invalid_name_raises(self):
        """Looking up an unknown or empty member name raises KeyError."""
        with self.assertRaises(KeyError):
            _ = ProfilerAction["INVALID_ACTION"]
        with self.assertRaises(KeyError):
            _ = ProfilerAction[""]

    def test_invalid_value_raises(self):
        """Constructing ProfilerAction from an invalid value raises ValueError."""
        with self.assertRaises(ValueError):
            ProfilerAction(-999)
        with self.assertRaises(ValueError):
            ProfilerAction("not_a_value")


if __name__ == "__main__":
    run_tests()
