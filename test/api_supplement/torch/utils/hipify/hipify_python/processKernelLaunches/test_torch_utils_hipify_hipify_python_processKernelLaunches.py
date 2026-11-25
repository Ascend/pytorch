# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.hipify.hipify_python.processKernelLaunches 接口功能正确性
API 名称：torch.utils.hipify.hipify_python.processKernelLaunches
API 签名：torch.utils.hipify.hipify_python.processKernelLaunches(string, stats)

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 空字符串与非空 CUDA 源码             | 已覆盖（empty_string、basic_kernel_launch）        |
| 枚举选项         | 枚举类参数选项                       | 未覆盖（无枚举类参数）                             |
| 参数类型         | 源码字符串与 stats 统计字典          | 已覆盖（basic_kernel_launch）                      |
| 传参与不传参     | 必填参数传入与缺参                   | 已覆盖（basic_kernel_launch、missing_args_raises） |
| 等价类/边界值    | 注释内核启动、普通代码与真实启动语法 | 已覆盖（commented_launch_ignored、                 |
|                  |                                      | non_kernel_unchanged、basic_kernel_launch）        |
| 正常传参场景     | 重写 <<<grid, block>>> 并更新 stats  | 已覆盖（basic_kernel_launch、non_kernel_unchanged）|
| 异常传参场景     | 缺失必填参数抛出 TypeError           | 已覆盖（missing_args_raises）                      |
| 混合设备类型     | 多 Tensor 输入 NPU/CPU 混合设备场景  | 未覆盖（纯源码字符串改写工具，无 Tensor 输入）     |

未覆盖项及原因：
- 枚举选项：该接口仅接收源码字符串与统计字典，无枚举类参数。
- 混合设备类型：该接口为纯源码文本转换函数，不涉及 Tensor 或设备迁移。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.utils.hipify.hipify_python import processKernelLaunches

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
class TestHipifyProcessKernelLaunches(TestCase):
    """Test cases for torch.utils.hipify.hipify_python.processKernelLaunches."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_basic_kernel_launch(self):
        """Rewrites CUDA <<<grid, block>>> launch to hipLaunchKernelGGL and updates stats."""
        stats = {"kernel_launches": []}
        src = "my_kernel<<<grid, block>>>(a, b);"
        out = processKernelLaunches(src, stats)
        expected = "hipLaunchKernelGGL((my_kernel), dim3(grid), dim3(block), 0, 0, a, b);"
        self.assertEqual(out, expected)
        self.assertEqual(len(stats["kernel_launches"]), 1)
        self.assertIn("hipLaunchKernelGGL", stats["kernel_launches"][0])

    def test_non_kernel_unchanged(self):
        """Source without CUDA kernel launches is returned unchanged."""
        stats = {"kernel_launches": []}
        src = "int x = foo(a, b);"
        out = processKernelLaunches(src, stats)
        self.assertEqual(out, src)
        self.assertEqual(stats["kernel_launches"], [])

    def test_commented_launch_ignored(self):
        """Kernel launches inside line comments are not rewritten."""
        stats = {"kernel_launches": []}
        src = "// my_kernel<<<grid, block>>>(a, b);\n"
        out = processKernelLaunches(src, stats)
        self.assertEqual(out, src)
        self.assertEqual(stats["kernel_launches"], [])

    def test_empty_string(self):
        """Empty source string returns empty string without modifying stats."""
        stats = {"kernel_launches": []}
        out = processKernelLaunches("", stats)
        self.assertEqual(out, "")
        self.assertEqual(stats["kernel_launches"], [])

    def test_missing_args_raises(self):
        """Calling processKernelLaunches with missing required args raises TypeError."""
        with self.assertRaises(TypeError):
            processKernelLaunches()
        with self.assertRaises(TypeError):
            processKernelLaunches("my_kernel<<<g, b>>>(a);")


if __name__ == "__main__":
    run_tests()
