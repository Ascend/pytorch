# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.profiler_util.StringTable.clear 接口功能正确性
API 名称：torch.autograd.profiler_util.StringTable.clear
API 签名：StringTable.clear(self) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------------------------------|--------------------------------------------------|
| 空/非空 | 空表与非空表执行清空 | 已覆盖（clear_empty_table、clear_populated_table） |
| 枚举选项 | 枚举参数各合法取值 | 未覆盖，该接口无枚举参数 |
| 参数类型 | 不同类型参数输入 | 未覆盖，该接口无参数（多余传参由 extra_arg_raises 覆盖） |
| 传参与不传参 | 无参正常调用与多余参数调用 | 已覆盖（clear_populated_table、extra_arg_raises） |
| 等价类/边界值 | 空表、重复清空幂等及清空后重新填充 | 已覆盖（clear_empty_table、clear_idempotent、repopulate_after_clear） |
| 正常传参场景 | 原地清空、保持对象标识及后续重建 | 已覆盖（clear_populated_table、clear_preserves_identity、repopulate_after_clear） |
| 异常传参场景 | 传入多余参数触发 TypeError | 已覆盖（extra_arg_raises） |
| 混合设备类型 | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖，该接口为纯 Python 字典方法，不涉及 Tensor 与设备迁移 |

未覆盖项及原因：
- 枚举选项：该接口无枚举参数。
- 参数类型：该接口不接受任何参数，非法传参已归入异常传参场景。
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
class TestStringTableClear(TestCase):
    """Test cases for torch.autograd.profiler_util.StringTable.clear."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_clear_populated_table(self):
        """clear() empties a populated StringTable and returns None."""
        table = StringTable()
        _ = table["a"]
        _ = table["aten::add"]
        self.assertEqual(len(table), 2)

        ret = table.clear()
        self.assertIsNone(ret)
        self.assertEqual(len(table), 0)
        self.assertNotIn("a", table)
        self.assertNotIn("aten::add", table)
        self.assertEqual(dict(table), {})

    def test_clear_preserves_identity(self):
        """clear() mutates the StringTable in place without changing object identity."""
        table = StringTable()
        alias = table
        _ = table["op_name"]
        self.assertEqual(len(alias), 1)

        table.clear()
        self.assertIs(table, alias)
        self.assertEqual(len(alias), 0)

    def test_clear_empty_table(self):
        """Calling clear() on an empty StringTable succeeds and returns None."""
        table = StringTable()
        self.assertEqual(len(table), 0)

        ret = table.clear()
        self.assertIsNone(ret)
        self.assertEqual(len(table), 0)

    def test_clear_idempotent(self):
        """Calling clear() multiple times leaves the table empty."""
        table = StringTable()
        table["k"] = "v"
        table.clear()
        table.clear()
        self.assertEqual(len(table), 0)

    def test_repopulate_after_clear(self):
        """After clear(), accessing missing keys repopulates via demangling."""
        table = StringTable()
        table["a"] = "custom_override"
        table.clear()
        self.assertNotIn("a", table)

        val_short = table["a"]
        self.assertEqual(val_short, "a")
        val_long = table["aten::matmul"]
        self.assertEqual(val_long, torch._C._demangle("aten::matmul"))
        self.assertEqual(len(table), 2)

    def test_extra_arg_raises(self):
        """Passing an unexpected argument to clear() raises TypeError."""
        table = StringTable()
        table["keep"] = "value"
        with self.assertRaises(TypeError):
            table.clear("unexpected")
        self.assertEqual(len(table), 1)
        self.assertEqual(table["keep"], "value")


if __name__ == "__main__":
    run_tests()
