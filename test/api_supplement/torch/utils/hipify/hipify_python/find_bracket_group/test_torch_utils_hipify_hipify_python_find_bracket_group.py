# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.hipify.hipify_python.find_bracket_group 接口功能正确性
API 名称：torch.utils.hipify.hipify_python.find_bracket_group
API 签名：torch.utils.hipify.hipify_python.find_bracket_group(input_string, start)

覆盖维度表：
| 覆盖维度      | 说明                                 | 覆盖情况                                             |
|---------------|--------------------------------------|------------------------------------------------------|
| 空/非空       | 空字符串/无括号串与非空大括号块      | 已覆盖（empty_and_no_match、simple_brackets）        |
| 枚举选项      | 枚举型参数选项                       | 未覆盖（固定匹配大括号 {}，无枚举选项）              |
| 参数类型      | str 文本与 int 起始偏移              | 已覆盖（simple_brackets、nonzero_start）             |
| 传参与不传参  | 完整传参与缺少必需参数               | 已覆盖（simple_brackets、missing_args_raises）       |
| 等价类/边界值 | 嵌套括号、未闭合、非零起点及越界起点 | 已覆盖（nested_brackets、unmatched_brackets、start_past_end） |
| 正常传参场景  | 定位外层大括号起止下标               | 已覆盖（simple_brackets、nested_brackets、nonzero_start） |
| 异常传参场景  | 缺参抛出 TypeError                   | 已覆盖（missing_args_raises）                        |
| 混合设备类型  | 多 Tensor 输入时 NPU/CPU 混合        | 未覆盖（纯字符串括号匹配工具，无 Tensor 输入）       |

未覆盖项及原因：
- 枚举选项：find_bracket_group 固定查找 {} 大括号对，无枚举类参数。
- 混合设备类型：接口为纯字符串索引解析函数，无 Tensor 输入，不涉及设备迁移。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.utils.hipify.hipify_python import find_bracket_group

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
class TestHipifyFindBracketGroup(TestCase):
    """Test cases for torch.utils.hipify.hipify_python.find_bracket_group."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_simple_brackets(self):
        """Finds start and end indices of a simple curly-brace block."""
        text = "void f() { return; }"
        start_idx, end_idx = find_bracket_group(text, 0)
        self.assertEqual((start_idx, end_idx), (9, 19))
        self.assertEqual(text[start_idx:end_idx + 1], "{ return; }")

    def test_nested_brackets(self):
        """Nested curly braces match the outermost closing brace."""
        text = "{ if (x) { y(); } }"
        start_idx, end_idx = find_bracket_group(text, 0)
        self.assertEqual((start_idx, end_idx), (0, len(text) - 1))

    def test_nonzero_start(self):
        """Non-zero start skips earlier brace groups and finds the next one."""
        text = "{first} {second}"
        start_idx, end_idx = find_bracket_group(text, 7)
        self.assertEqual(text[start_idx:end_idx + 1], "{second}")

    def test_unmatched_brackets(self):
        """Unclosed curly brace returns (None, None)."""
        self.assertEqual(find_bracket_group("{ unclosed { inner }", 0), (None, None))

    def test_empty_and_no_match(self):
        """Empty string or string without curly braces returns (None, None)."""
        self.assertEqual(find_bracket_group("", 0), (None, None))
        self.assertEqual(find_bracket_group("no braces here", 0), (None, None))

    def test_start_past_end(self):
        """Start index beyond string length returns (None, None)."""
        self.assertEqual(find_bracket_group("{a}", 10), (None, None))

    def test_missing_args_raises(self):
        """Calling find_bracket_group with missing arguments raises TypeError."""
        with self.assertRaises(TypeError):
            find_bracket_group()
        with self.assertRaises(TypeError):
            find_bracket_group("{}")


if __name__ == "__main__":
    run_tests()
