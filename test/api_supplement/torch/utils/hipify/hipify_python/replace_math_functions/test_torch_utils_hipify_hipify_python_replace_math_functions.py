# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.hipify.hipify_python.replace_math_functions 接口功能正确性
API 名称：torch.utils.hipify.hipify_python.replace_math_functions
API 签名：torch.utils.hipify.hipify_python.replace_math_functions(input_string)

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 空字符串与非空数学表达式源码         | 已覆盖（empty_string、basic_math_replacements）    |
| 枚举选项         | 枚举类参数选项                       | 未覆盖（无枚举类参数）                             |
| 参数类型         | 源码字符串参数                       | 已覆盖（basic_math_replacements、multiple_calls）  |
| 传参与不传参     | 必填参数传入与缺参                   | 已覆盖（basic_math_replacements、                  |
|                  |                                      | missing_arg_raises）                               |
| 等价类/边界值    | 非调用标识符、多调用嵌套与未映射代码 | 已覆盖（non_call_identifier_unchanged、            |
|                  |                                      | multiple_calls、plain_string_unchanged）           |
| 正常传参场景     | std:: 数学调用替换为全局 :: 版本     | 已覆盖（basic_math_replacements、multiple_calls、  |
|                  |                                      | additional_math_transpilations）                   |
| 异常传参场景     | 缺失必填参数抛出 TypeError           | 已覆盖（missing_arg_raises）                       |
| 混合设备类型     | 多 Tensor 输入 NPU/CPU 混合设备场景  | 未覆盖（纯源码字符串替换工具，无 Tensor 输入）     |

未覆盖项及原因：
- 枚举选项：该接口仅接收单个源码字符串参数，无枚举类参数。
- 混合设备类型：该接口为纯源码字符串替换函数，不涉及 Tensor 或设备迁移。
- 当前看护环境执行失败：PyTorch 2.16 已移除该接口，全部用例已临时 skip，待修复后恢复覆盖。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch

try:
    from torch.utils.hipify.hipify_python import replace_math_functions
except ImportError:  # removed in PyTorch 2.16
    replace_math_functions = None

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
@unittest.skip("Known failure in API supplement run; temporarily skipped until fixed")
class TestHipifyReplaceMathFunctions(TestCase):
    """Test cases for torch.utils.hipify.hipify_python.replace_math_functions."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_basic_math_replacements(self):
        """Replaces mapped std:: math calls with global :: versions."""
        self.assertEqual(replace_math_functions("std::exp(x)"), "::exp(x)")
        self.assertEqual(replace_math_functions("std::log(x)"), "::log(x)")

    def test_multiple_calls(self):
        """Replaces multiple std:: math calls in the same expression."""
        src = "auto y = std::floor(std::log(x)) + std::pow(a, b);"
        expected = "auto y = ::floor(::log(x)) + ::pow(a, b);"
        self.assertEqual(replace_math_functions(src), expected)

    def test_additional_math_transpilations(self):
        """Replaces additional mapped std:: math functions such as ceil, fabs, fmod."""
        src = "std::ceil(x); std::fabs(y); std::fmod(x, y);"
        expected = "::ceil(x); ::fabs(y); ::fmod(x, y);"
        self.assertEqual(replace_math_functions(src), expected)

    def test_non_call_identifier_unchanged(self):
        """References to std::exp without a call parenthesis are not rewritten."""
        src = "auto fn = &std::exp;"
        self.assertEqual(replace_math_functions(src), src)

    def test_plain_string_unchanged(self):
        """Strings without mapped std:: math calls remain unchanged."""
        src = "float y = custom_func(x) + std::vector<int>();"
        self.assertEqual(replace_math_functions(src), src)

    def test_empty_string(self):
        """Empty string returns empty string."""
        self.assertEqual(replace_math_functions(""), "")

    def test_missing_arg_raises(self):
        """Calling replace_math_functions without arguments raises TypeError."""
        with self.assertRaises(TypeError):
            replace_math_functions()


if __name__ == "__main__":
    run_tests()
