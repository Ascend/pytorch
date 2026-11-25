# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.hipify.hipify_python.fix_static_global_kernels 接口功能正确性
API 名称：torch.utils.hipify.hipify_python.fix_static_global_kernels
API 签名：torch.utils.hipify.hipify_python.fix_static_global_kernels(in_txt)

覆盖维度表：
| 覆盖维度      | 说明                                 | 覆盖情况                                              |
|---------------|--------------------------------------|-------------------------------------------------------|
| 空/非空       | 空字符串与非空 CUDA 核源码文本       | 已覆盖（empty_string、basic_replace）                 |
| 枚举选项      | 枚举型参数选项                       | 未覆盖（仅接收源码字符串 in_txt，无枚举选项）         |
| 参数类型      | 合法 str 与非法 None 输入            | 已覆盖（basic_replace、invalid_type_raises）          |
| 传参与不传参  | 传入 in_txt 与缺参调用               | 已覆盖（basic_replace、missing_arg_raises）           |
| 等价类/边界值 | 多处出现、无匹配与缺少前导空格边界   | 已覆盖（multiple_occurrences、unchanged_when_no_match、without_leading_space_unchanged） |
| 正常传参场景  | 将 ' __global__ static' 替换为目标串 | 已覆盖（basic_replace、multiple_occurrences、unchanged_when_no_match） |
| 异常传参场景  | 缺参 TypeError 与非 str 类型异常     | 已覆盖（missing_arg_raises、invalid_type_raises）     |
| 混合设备类型  | 多 Tensor 输入时 NPU/CPU 混合        | 未覆盖（纯源码文本替换函数，无 Tensor 输入）          |

未覆盖项及原因：
- 枚举选项：接口仅接收单个字符串参数 in_txt，无枚举类参数。
- 混合设备类型：接口仅对 CUDA 源码文本执行字符串替换，无 Tensor 输入，不涉及设备迁移。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.utils.hipify.hipify_python import fix_static_global_kernels

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
class TestHipifyFixStaticGlobalKernels(TestCase):
    """Test cases for torch.utils.hipify.hipify_python.fix_static_global_kernels."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_basic_replace(self):
        """Replaces ' __global__ static' with '__global__'."""
        src = "template<typename T>\n __global__ static void foo(T* x) {}\n"
        expected = "template<typename T>\n__global__ void foo(T* x) {}\n"
        self.assertEqual(fix_static_global_kernels(src), expected)

    def test_multiple_occurrences(self):
        """All occurrences in a multi-kernel source buffer are fixed."""
        src = " __global__ static void k1() {}\n __global__ static void k2() {}\n"
        expected = "__global__ void k1() {}\n__global__ void k2() {}\n"
        self.assertEqual(fix_static_global_kernels(src), expected)

    def test_unchanged_when_no_match(self):
        """Code without ' __global__ static' is returned unchanged."""
        src = "__global__ void bar(float* p) { p[0] = 1.0f; }\n"
        self.assertEqual(fix_static_global_kernels(src), src)

    def test_empty_string(self):
        """Empty string returns empty string."""
        self.assertEqual(fix_static_global_kernels(""), "")

    def test_without_leading_space_unchanged(self):
        """Pattern without leading space before __global__ is not modified."""
        src = "__global__ static void foo() {}\n"
        self.assertEqual(fix_static_global_kernels(src), src)

    def test_missing_arg_raises(self):
        """Calling fix_static_global_kernels without arguments raises TypeError."""
        with self.assertRaises(TypeError):
            fix_static_global_kernels()

    def test_invalid_type_raises(self):
        """Passing non-string input raises AttributeError."""
        with self.assertRaises(AttributeError):
            fix_static_global_kernels(None)


if __name__ == "__main__":
    run_tests()
