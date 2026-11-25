# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.profiler_util.StringTable.setdefault 接口功能正确性
API 名称：torch.autograd.profiler_util.StringTable.setdefault
API 签名：StringTable.setdefault(self, key, default=None)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------------------------------|--------------------------------------------------|
| 空/非空 | 空表插入新键与非空表访问已有键 | 已覆盖（absent_key_with_default、existing_key_not_overwritten） |
| 枚举选项 | 枚举参数各合法取值 | 未覆盖，该接口无枚举参数 |
| 参数类型 | 可变对象默认值、None 默认值及不可哈希键 | 已覆盖（absent_key_with_default、absent_key_default_omitted、unhashable_key_raises） |
| 传参与不传参 | default 参数传入与省略 | 已覆盖（absent_key_with_default、absent_key_default_omitted） |
| 等价类/边界值 | 键不存在/已存在及不触发 __missing__ 边界 | 已覆盖（existing_key_not_overwritten、does_not_trigger_missing_demangle） |
| 正常传参场景 | 正常写入默认值与返回已有值 | 已覆盖（absent_key_with_default、absent_key_default_omitted、existing_key_not_overwritten） |
| 异常传参场景 | 参数个数错误与不可哈希键触发 TypeError | 已覆盖（wrong_arity_raises、unhashable_key_raises） |
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
from unittest.mock import patch

import torch
from torch.autograd.profiler_util import StringTable

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


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestStringTableSetdefault(TestCase):
    """Test cases for torch.autograd.profiler_util.StringTable.setdefault."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_absent_key_with_default(self):
        """On absent key, setdefault inserts and returns the exact default object."""
        table = StringTable()
        sentinel = ["custom_value"]
        ret = table.setdefault("new_key", sentinel)
        self.assertIs(ret, sentinel)
        self.assertIn("new_key", table)
        self.assertIs(table["new_key"], sentinel)
        self.assertEqual(len(table), 1)

    def test_absent_key_default_omitted(self):
        """When default is omitted, setdefault inserts None for an absent key."""
        table = StringTable()
        ret = table.setdefault("omitted_key")
        self.assertIsNone(ret)
        self.assertIn("omitted_key", table)
        self.assertIsNone(table["omitted_key"])

    def test_existing_key_not_overwritten(self):
        """On existing key, setdefault returns existing value without overwriting."""
        table = StringTable()
        original = table["aten::mm"]
        ret = table.setdefault("aten::mm", "should_not_replace")
        self.assertEqual(ret, original)
        self.assertEqual(table["aten::mm"], original)

    def test_does_not_trigger_missing_demangle(self):
        """setdefault on an absent key does not invoke __missing__ or _demangle."""
        table = StringTable()
        with patch.object(torch._C, "_demangle", side_effect=AssertionError("should not be called")):
            ret = table.setdefault("_Z3fooi", "manual_entry")
        self.assertEqual(ret, "manual_entry")
        self.assertEqual(table["_Z3fooi"], "manual_entry")

    def test_wrong_arity_raises(self):
        """Calling setdefault with 0 args or >2 args raises TypeError."""
        table = StringTable()
        with self.assertRaises(TypeError):
            table.setdefault()
        with self.assertRaises(TypeError):
            table.setdefault("k", "v", "extra")
        self.assertEqual(len(table), 0)

    def test_unhashable_key_raises(self):
        """Passing an unhashable key raises TypeError without polluting table."""
        table = StringTable()
        with self.assertRaises(TypeError):
            table.setdefault(["unhashable"], "val")
        self.assertEqual(len(table), 0)


if __name__ == "__main__":
    run_tests()
