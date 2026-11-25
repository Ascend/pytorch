# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.hipify.hipify_python.find_closure_group 接口功能正确性
API 名称：torch.utils.hipify.hipify_python.find_closure_group
API 签名：torch.utils.hipify.hipify_python.find_closure_group(input_string, start, group)

覆盖维度表：
| 覆盖维度      | 说明                                 | 覆盖情况                                              |
|---------------|--------------------------------------|-------------------------------------------------------|
| 空/非空       | 空字符串/无匹配与非空闭合块          | 已覆盖（unmatched_and_empty、parentheses_group）      |
| 枚举选项      | ()、<>、[] 等不同开闭分隔符对        | 已覆盖（parentheses_group、angle_and_square_groups）  |
| 参数类型      | str 文本、int 起点与 list 分隔符对   | 已覆盖（parentheses_group、angle_and_square_groups）  |
| 传参与不传参  | 完整传参与缺少必需参数               | 已覆盖（parentheses_group、missing_args_raises）      |
| 等价类/边界值 | 多层嵌套、非零起点与未闭合边界       | 已覆盖（nested_group、nonzero_start、unmatched_and_empty） |
| 正常传参场景  | 查找各类成对分隔符的外层起止下标     | 已覆盖（parentheses_group、angle_and_square_groups、nested_group） |
| 异常传参场景  | 缺参抛出 TypeError                   | 已覆盖（missing_args_raises）                         |
| 混合设备类型  | 多 Tensor 输入时 NPU/CPU 混合        | 未覆盖（纯字符串闭合符号匹配工具，无 Tensor 输入）    |

未覆盖项及原因：
- 混合设备类型：接口为纯字符串闭合分组解析函数，无 Tensor 输入，不涉及设备迁移。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.utils.hipify.hipify_python import find_closure_group

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
class TestHipifyFindClosureGroup(TestCase):
    """Test cases for torch.utils.hipify.hipify_python.find_closure_group."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_parentheses_group(self):
        """Finds balanced parentheses starting from index 0."""
        self.assertEqual(find_closure_group("(hi)", 0, ["(", ")"]), (0, 3))

    def test_angle_and_square_groups(self):
        """Supports custom delimiter pairs such as '<'/'>' and '['/']'."""
        tmpl = "vec<pair<int, float>> x"
        start_idx, end_idx = find_closure_group(tmpl, 0, ["<", ">"])
        self.assertEqual(tmpl[start_idx:end_idx + 1], "<pair<int, float>>")

        arr = "a[i + b[j]]"
        s_idx, e_idx = find_closure_group(arr, 0, ["[", "]"])
        self.assertEqual(arr[s_idx:e_idx + 1], "[i + b[j]]")

    def test_nested_group(self):
        """Deeply nested delimiters return outer open/close indices."""
        text = "fn((a + (b * c)))"
        start_idx, end_idx = find_closure_group(text, 0, ["(", ")"])
        self.assertEqual((start_idx, end_idx), (2, len(text) - 1))

    def test_nonzero_start(self):
        """Non-zero start skips earlier balanced closures."""
        text = "(one) (two)"
        start_idx, end_idx = find_closure_group(text, 5, ["(", ")"])
        self.assertEqual(text[start_idx:end_idx + 1], "(two)")

    def test_unmatched_and_empty(self):
        """Unmatched delimiters or empty input return (None, None)."""
        self.assertEqual(find_closure_group("(a + (b)", 0, ["(", ")"]), (None, None))
        self.assertEqual(find_closure_group("", 0, ["(", ")"]), (None, None))
        self.assertEqual(find_closure_group("plain text", 0, ["{", "}"]), (None, None))

    def test_missing_args_raises(self):
        """Calling find_closure_group with missing arguments raises TypeError."""
        with self.assertRaises(TypeError):
            find_closure_group()
        with self.assertRaises(TypeError):
            find_closure_group("(a)", 0)


if __name__ == "__main__":
    run_tests()
