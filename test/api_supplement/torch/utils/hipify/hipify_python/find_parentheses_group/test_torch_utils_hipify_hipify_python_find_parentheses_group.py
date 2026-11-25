# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.hipify.hipify_python.find_parentheses_group 接口功能正确性
API 名称：torch.utils.hipify.hipify_python.find_parentheses_group
API 签名：torch.utils.hipify.hipify_python.find_parentheses_group(input_string, start)

覆盖维度表：
| 覆盖维度      | 说明                                 | 覆盖情况                                            |
|---------------|--------------------------------------|-----------------------------------------------------|
| 空/非空       | 空字符串/无括号串与非空圆括号串      | 已覆盖（empty_and_no_match、simple_parens）         |
| 枚举选项      | 枚举型参数选项                       | 未覆盖（固定匹配圆括号 ()，无枚举选项）             |
| 参数类型      | str 文本与 int 起始偏移              | 已覆盖（simple_parens、nonzero_start）              |
| 传参与不传参  | 完整传参与缺少必需参数               | 已覆盖（simple_parens、missing_args_raises）        |
| 等价类/边界值 | 嵌套圆括号、非零起点与未闭合圆括号   | 已覆盖（nested_parens、nonzero_start、unmatched_parens） |
| 正常传参场景  | 定位外层圆括号起止下标               | 已覆盖（simple_parens、nested_parens、nonzero_start） |
| 异常传参场景  | 缺参抛出 TypeError                   | 已覆盖（missing_args_raises）                       |
| 混合设备类型  | 多 Tensor 输入时 NPU/CPU 混合        | 未覆盖（纯字符串圆括号匹配工具，无 Tensor 输入）    |

未覆盖项及原因：
- 枚举选项：find_parentheses_group 固定查找 () 圆括号对，无枚举类参数。
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
from torch.utils.hipify.hipify_python import find_parentheses_group

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
class TestHipifyFindParenthesesGroup(TestCase):
    """Test cases for torch.utils.hipify.hipify_python.find_parentheses_group."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_simple_parens(self):
        """Finds start and end indices of a simple parentheses group."""
        text = "func(a, b)"
        start_idx, end_idx = find_parentheses_group(text, 0)
        self.assertEqual((start_idx, end_idx), (4, 9))
        self.assertEqual(text[start_idx:end_idx + 1], "(a, b)")

    def test_nested_parens(self):
        """Nested parentheses match the outermost closing parenthesis."""
        text = "call(foo(1, 2), bar(3))"
        start_idx, end_idx = find_parentheses_group(text, 0)
        self.assertEqual((start_idx, end_idx), (4, len(text) - 1))
        self.assertEqual(text[start_idx:end_idx + 1], "(foo(1, 2), bar(3))")

    def test_nonzero_start(self):
        """Non-zero start skips earlier groups and matches the subsequent group."""
        text = "(first) (second)"
        start_idx, end_idx = find_parentheses_group(text, 7)
        self.assertEqual(text[start_idx:end_idx + 1], "(second)")

    def test_unmatched_parens(self):
        """Unclosed parenthesis returns (None, None)."""
        self.assertEqual(find_parentheses_group("fn(a, (b)", 0), (None, None))

    def test_empty_and_no_match(self):
        """Empty string or text without parentheses returns (None, None)."""
        self.assertEqual(find_parentheses_group("", 0), (None, None))
        self.assertEqual(find_parentheses_group("no parens", 0), (None, None))

    def test_missing_args_raises(self):
        """Calling find_parentheses_group with missing arguments raises TypeError."""
        with self.assertRaises(TypeError):
            find_parentheses_group()
        with self.assertRaises(TypeError):
            find_parentheses_group("()")


if __name__ == "__main__":
    run_tests()
