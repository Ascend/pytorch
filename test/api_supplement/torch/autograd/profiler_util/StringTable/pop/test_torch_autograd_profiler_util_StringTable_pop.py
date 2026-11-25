# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.profiler_util.StringTable.pop 接口功能正确性
API 名称：torch.autograd.profiler_util.StringTable.pop
API 签名：StringTable.pop(self, key, default=...)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------------------------------|--------------------------------------------------|
| 空/非空 | 空表与非空表弹出键值 | 已覆盖（pop_empty_table、pop_existing_key） |
| 枚举选项 | 枚举参数各合法取值 | 未覆盖，该接口无枚举参数 |
| 参数类型 | 哨兵对象 default 及不可哈希键类型 | 已覆盖（pop_absent_with_default、unhashable_key_raises） |
| 传参与不传参 | default 参数省略与显式传入 | 已覆盖（pop_existing_key、pop_existing_ignores_default、pop_absent_with_default） |
| 等价类/边界值 | 键存在/不存在及空表边界 | 已覆盖（pop_empty_table、pop_absent_with_default、pop_absent_no_default_raises） |
| 正常传参场景 | 正常弹出存在键及带默认值弹出缺失键 | 已覆盖（pop_existing_key、pop_existing_ignores_default、pop_absent_with_default） |
| 异常传参场景 | 缺失键无默认值、参数个数错误、不可哈希键 | 已覆盖（pop_absent_no_default_raises、wrong_arity_raises、unhashable_key_raises） |
| 混合设备类型 | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖，该接口为纯 Python 字典方法，不涉及 Tensor 与设备迁移 |

未覆盖项及原因：
- 枚举选项：该接口无枚举参数。
- 混合设备类型：该接口为纯 Python 映射操作，不涉及 Tensor 输入。

注意：对可在 CPU/NPU 两侧执行的非图模式浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.autograd.profiler_util import StringTable

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
class TestStringTablePop(TestCase):
    """Test cases for torch.autograd.profiler_util.StringTable.pop."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_pop_existing_key(self):
        """pop() removes an existing key and returns its stored value."""
        table = StringTable()
        expected = table["aten::add"]
        _ = table["x"]
        self.assertEqual(len(table), 2)
        popped = table.pop("aten::add")
        self.assertEqual(popped, expected)
        self.assertNotIn("aten::add", table)
        self.assertEqual(len(table), 1)
        self.assertIn("x", table)

    def test_pop_existing_ignores_default(self):
        """pop() returns the stored value and ignores default when key exists."""
        table = StringTable()
        table["k"] = "stored_val"
        self.assertEqual(table.pop("k", "unused_default"), "stored_val")
        self.assertNotIn("k", table)

    def test_pop_absent_with_default(self):
        """pop() returns the provided default object when key is absent."""
        table = StringTable()
        _ = table["present"]
        sentinel = object()
        self.assertIs(table.pop("missing_key", sentinel), sentinel)
        self.assertNotIn("missing_key", table)
        self.assertEqual(len(table), 1)

    def test_pop_absent_no_default_raises(self):
        """pop() raises KeyError for an absent key when no default is given."""
        table = StringTable()
        _ = table["present"]
        with self.assertRaises(KeyError):
            table.pop("absent_key")
        self.assertNotIn("absent_key", table)
        self.assertEqual(len(table), 1)

    def test_pop_empty_table(self):
        """pop() on an empty table returns default or raises KeyError without default."""
        table = StringTable()
        self.assertIsNone(table.pop("none_key", None))
        self.assertEqual(len(table), 0)
        with self.assertRaises(KeyError):
            table.pop("none_key")
        self.assertEqual(len(table), 0)

    def test_wrong_arity_raises(self):
        """Calling pop() with 0 args or >2 args raises TypeError."""
        table = StringTable()
        table["a"] = "val"
        with self.assertRaises(TypeError):
            table.pop()
        with self.assertRaises(TypeError):
            table.pop("a", "d1", "d2")
        self.assertEqual(table["a"], "val")

    def test_unhashable_key_raises(self):
        """Passing an unhashable key to pop() raises TypeError."""
        table = StringTable()
        table["ok"] = "v"
        with self.assertRaises(TypeError):
            table.pop(["bad"])
        self.assertEqual(len(table), 1)


if __name__ == "__main__":
    run_tests()
