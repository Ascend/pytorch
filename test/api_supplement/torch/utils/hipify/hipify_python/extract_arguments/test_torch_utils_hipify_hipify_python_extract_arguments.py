# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.hipify.hipify_python.extract_arguments 接口功能正确性
API 名称：torch.utils.hipify.hipify_python.extract_arguments
API 签名：torch.utils.hipify.hipify_python.extract_arguments(start, string)

覆盖维度表：
| 覆盖维度      | 说明                                 | 覆盖情况                                              |
|---------------|--------------------------------------|-------------------------------------------------------|
| 空/非空       | 空括号 () 与非空参数列表             | 已覆盖（empty_parens、simple_args）                   |
| 枚举选项      | 枚举型参数选项                       | 未覆盖（参数为偏移整数与源码字符串，无枚举选项）      |
| 参数类型      | int 起始索引与 str 源码文本          | 已覆盖（simple_args、nonzero_start）                  |
| 传参与不传参  | 完整传参与缺少必需参数               | 已覆盖（simple_args、missing_args_raises）            |
| 等价类/边界值 | 单参数、空括号、嵌套模板与非零起点   | 已覆盖（single_arg、empty_parens、nested_parens_and_templates、nonzero_start） |
| 正常传参场景  | 提取括号内各参数起止区间             | 已覆盖（simple_args、single_arg、nested_parens_and_templates、nonzero_start） |
| 异常传参场景  | 缺参抛出 TypeError                   | 已覆盖（missing_args_raises）                         |
| 混合设备类型  | 多 Tensor 输入时 NPU/CPU 混合        | 未覆盖（纯字符串语法解析工具，无 Tensor 输入）        |

未覆盖项及原因：
- 枚举选项：参数 start 与 string 分别为整数索引和字符串，无枚举类参数。
- 混合设备类型：接口为纯字符串参数切分函数，无 Tensor 输入，不涉及设备迁移。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.utils.hipify.hipify_python import extract_arguments

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
class TestHipifyExtractArguments(TestCase):
    """Test cases for torch.utils.hipify.hipify_python.extract_arguments."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_simple_args(self):
        """Extracts start/end spans for simple comma-separated arguments."""
        text = "(a, b, c)"
        args = extract_arguments(0, text)
        self.assertEqual(len(args), 3)
        extracted = [text[item["start"]:item["end"]].strip() for item in args]
        self.assertEqual(extracted, ["a", "b", "c"])

    def test_single_arg(self):
        """Single argument inside parentheses returns a one-element list."""
        text = "(only_arg)"
        args = extract_arguments(0, text)
        self.assertEqual(args, [{"start": 1, "end": 9}])
        self.assertEqual(text[args[0]["start"]:args[0]["end"]], "only_arg")

    def test_empty_parens(self):
        """Empty parentheses yields a single empty span."""
        text = "()"
        args = extract_arguments(0, text)
        self.assertEqual(args, [{"start": 1, "end": 1}])

    def test_nested_parens_and_templates(self):
        """Nested parentheses, template angle brackets, and '->' are respected."""
        text = "(foo(1, 2), bar<int, float>(ptr->val), 0)"
        args = extract_arguments(0, text)
        self.assertEqual(len(args), 3)
        extracted = [text[item["start"]:item["end"]].strip() for item in args]
        self.assertEqual(extracted, ["foo(1, 2)", "bar<int, float>(ptr->val)", "0"])

    def test_nonzero_start(self):
        """Non-zero start index parses parentheses beginning at that offset."""
        text = "my_fn(blocks, threads)"
        start_idx = text.index("(")
        args = extract_arguments(start_idx, text)
        extracted = [text[item["start"]:item["end"]].strip() for item in args]
        self.assertEqual(extracted, ["blocks", "threads"])

    def test_missing_args_raises(self):
        """Calling extract_arguments with missing arguments raises TypeError."""
        with self.assertRaises(TypeError):
            extract_arguments()
        with self.assertRaises(TypeError):
            extract_arguments(0)


if __name__ == "__main__":
    run_tests()
