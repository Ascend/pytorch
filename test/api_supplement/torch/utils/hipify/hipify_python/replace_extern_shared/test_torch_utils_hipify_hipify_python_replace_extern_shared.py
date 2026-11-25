# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.hipify.hipify_python.replace_extern_shared 接口功能正确性
API 名称：torch.utils.hipify.hipify_python.replace_extern_shared
API 签名：torch.utils.hipify.hipify_python.replace_extern_shared(input_string)

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 空字符串与非空 CUDA 共享内存声明     | 已覆盖（empty_string、basic_extern_shared）        |
| 枚举选项         | 枚举类参数选项                       | 未覆盖（无枚举类参数）                             |
| 参数类型         | 源码字符串参数                       | 已覆盖（basic_extern_shared、multi_word_type）     |
| 传参与不传参     | 必填参数传入与缺参                   | 已覆盖（basic_extern_shared、missing_arg_raises）  |
| 等价类/边界值    | 多词类型、多条声明与定长共享数组边界 | 已覆盖（multi_word_type、                          |
|                  |                                      | multiple_declarations_in_block、                   |
|                  |                                      | non_matching_unchanged）                           |
| 正常传参场景     | extern __shared__ 替换为动态共享宏   | 已覆盖（basic_extern_shared、multi_word_type、     |
|                  |                                      | multiple_declarations_in_block）                   |
| 异常传参场景     | 缺失必填参数抛出 TypeError           | 已覆盖（missing_arg_raises）                       |
| 混合设备类型     | 多 Tensor 输入 NPU/CPU 混合设备场景  | 未覆盖（纯源码字符串替换工具，无 Tensor 输入）     |

未覆盖项及原因：
- 枚举选项：该接口仅接收单个源码字符串参数，无枚举类参数。
- 混合设备类型：该接口为纯源码正则替换函数，不涉及 Tensor 或设备迁移。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.utils.hipify.hipify_python import replace_extern_shared

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
class TestHipifyReplaceExternShared(TestCase):
    """Test cases for torch.utils.hipify.hipify_python.replace_extern_shared."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_basic_extern_shared(self):
        """Replaces extern __shared__ float buf[]; with HIP_DYNAMIC_SHARED macro."""
        out = replace_extern_shared("extern __shared__ float buf[];")
        self.assertEqual(out, "HIP_DYNAMIC_SHARED( float, buf)")

    def test_multi_word_type(self):
        """Handles multi-word element types such as unsigned char."""
        out = replace_extern_shared("extern __shared__ unsigned char smem[];")
        self.assertEqual(out, "HIP_DYNAMIC_SHARED( unsigned char, smem)")

    def test_multiple_declarations_in_block(self):
        """Replaces multiple extern __shared__ declarations while keeping surrounding code."""
        src = (
            "__global__ void k() {\n"
            "  extern __shared__ float a[];\n"
            "  extern __shared__ int b[];\n"
            "}\n"
        )
        out = replace_extern_shared(src)
        self.assertIn("HIP_DYNAMIC_SHARED( float, a)", out)
        self.assertIn("HIP_DYNAMIC_SHARED( int, b)", out)
        self.assertIn("__global__ void k()", out)

    def test_non_matching_unchanged(self):
        """Regular shared arrays with fixed size or plain code remain unchanged."""
        src1 = "__shared__ float buf[128];"
        src2 = "int x = 42;"
        self.assertEqual(replace_extern_shared(src1), src1)
        self.assertEqual(replace_extern_shared(src2), src2)

    def test_empty_string(self):
        """Empty string returns empty string."""
        self.assertEqual(replace_extern_shared(""), "")

    def test_missing_arg_raises(self):
        """Calling replace_extern_shared without arguments raises TypeError."""
        with self.assertRaises(TypeError):
            replace_extern_shared()


if __name__ == "__main__":
    run_tests()
