# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.profiler_util.StringTable.popitem 接口功能正确性
API 名称：torch.autograd.profiler_util.StringTable.popitem
API 签名：StringTable.popitem(self) -> tuple

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------------------------------|--------------------------------------------------|
| 空/非空 | 空表与非空表弹出项 | 已覆盖（popitem_empty_raises、popitem_lifo_order、popitem_until_empty） |
| 枚举选项 | 枚举参数各合法取值 | 未覆盖，该接口无枚举参数 |
| 参数类型 | 不同类型参数输入 | 未覆盖，该接口无参数（多余传参由 extra_arg_raises 覆盖） |
| 传参与不传参 | 无参正常调用与多余参数调用 | 已覆盖（popitem_lifo_order、extra_arg_raises） |
| 等价类/边界值 | 弹出至空后继续弹出与初始空表 | 已覆盖（popitem_until_empty、popitem_empty_raises） |
| 正常传参场景 | LIFO 顺序弹出并移除键值对 | 已覆盖（popitem_lifo_order、popitem_until_empty） |
| 异常传参场景 | 空表抛 KeyError 及多余参数抛 TypeError | 已覆盖（popitem_empty_raises、extra_arg_raises） |
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
class TestStringTablePopitem(TestCase):
    """Test cases for torch.autograd.profiler_util.StringTable.popitem."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_popitem_lifo_order(self):
        """popitem() removes and returns (key, value) pairs in LIFO order."""
        table = StringTable()
        v1 = table["first_op"]
        v2 = table["second_op"]
        v3 = table["z"]
        self.assertEqual(len(table), 3)

        item3 = table.popitem()
        self.assertIsInstance(item3, tuple)
        self.assertEqual(item3, ("z", v3))
        self.assertEqual(len(table), 2)
        self.assertNotIn("z", table)

        item2 = table.popitem()
        self.assertEqual(item2, ("second_op", v2))
        self.assertEqual(len(table), 1)
        self.assertNotIn("second_op", table)

        item1 = table.popitem()
        self.assertEqual(item1, ("first_op", v1))
        self.assertEqual(len(table), 0)

    def test_popitem_until_empty(self):
        """Popping the last item empties the table; a further call raises KeyError."""
        table = StringTable()
        table["only"] = "val"

        key, val = table.popitem()
        self.assertEqual((key, val), ("only", "val"))
        self.assertEqual(len(table), 0)

        with self.assertRaises(KeyError):
            table.popitem()

    def test_popitem_empty_raises(self):
        """Calling popitem() on an empty StringTable raises KeyError."""
        table = StringTable()
        with self.assertRaises(KeyError):
            table.popitem()
        self.assertEqual(len(table), 0)

    def test_extra_arg_raises(self):
        """Passing any argument to popitem() raises TypeError without mutating table."""
        table = StringTable()
        table["keep"] = "v"
        with self.assertRaises(TypeError):
            table.popitem("keep")
        self.assertEqual(len(table), 1)
        self.assertEqual(table["keep"], "v")


if __name__ == "__main__":
    run_tests()
