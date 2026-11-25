# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.hipify.hipify_python.compute_stats 接口功能正确性
API 名称：torch.utils.hipify.hipify_python.compute_stats
API 签名：torch.utils.hipify.hipify_python.compute_stats(stats) -> None

覆盖维度表：
| 覆盖维度      | 说明                                 | 覆盖情况                                             |
|---------------|--------------------------------------|------------------------------------------------------|
| 空/非空       | 空统计列表与非空统计字典             | 已覆盖（empty_stats、populated_stats）               |
| 枚举选项      | 枚举型参数选项                       | 未覆盖（仅接收统计字典 stats，无枚举选项）           |
| 参数类型      | 标准统计字典结构与返回值 None        | 已覆盖（populated_stats、empty_stats）               |
| 传参与不传参  | 传入 stats 与省略参数                | 已覆盖（populated_stats、missing_arg_raises）        |
| 等价类/边界值 | 空记录与重复 unsupported_calls 去重  | 已覆盖（empty_stats、deduplicates_unsupported_calls） |
| 正常传参场景  | 打印不支持调用计数与核启动替换计数   | 已覆盖（populated_stats、empty_stats、deduplicates_unsupported_calls） |
| 异常传参场景  | 缺失必需键 KeyError 与缺参 TypeError | 已覆盖（missing_key_raises、missing_arg_raises）     |
| 混合设备类型  | 多 Tensor 输入时 NPU/CPU 混合        | 未覆盖（纯主机侧标准输出统计函数，无 Tensor 输入）   |

未覆盖项及原因：
- 枚举选项：接口仅接收单个 stats 字典参数，无枚举类参数。
- 混合设备类型：接口仅汇总并打印 hipify 统计信息，无 Tensor 输入，不涉及设备迁移。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import contextlib
import io
import os
import unittest

import torch
from torch.utils.hipify.hipify_python import compute_stats

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
class TestHipifyComputeStats(TestCase):
    """Test cases for torch.utils.hipify.hipify_python.compute_stats."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_populated_stats(self):
        """Prints counts and names for populated unsupported calls and kernel launches."""
        stats = {
            "unsupported_calls": [("cudaFuncA", "a.cu"), ("cudaFuncB", "b.cu")],
            "kernel_launches": ["hipLaunchKernelGGL(k1)", "hipLaunchKernelGGL(k2)", "hipLaunchKernelGGL(k3)"],
        }
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ret = compute_stats(stats)
        output = buf.getvalue()
        self.assertIsNone(ret)
        self.assertIn("Total number of unsupported CUDA function calls: 2", output)
        self.assertIn("cudaFuncA", output)
        self.assertIn("cudaFuncB", output)
        self.assertIn("Total number of replaced kernel launches: 3", output)

    def test_empty_stats(self):
        """Prints zero totals for empty unsupported_calls and kernel_launches."""
        stats = {"unsupported_calls": [], "kernel_launches": []}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ret = compute_stats(stats)
        output = buf.getvalue()
        self.assertIsNone(ret)
        self.assertIn("Total number of unsupported CUDA function calls: 0", output)
        self.assertIn("Total number of replaced kernel launches: 0", output)

    def test_deduplicates_unsupported_calls(self):
        """Duplicate unsupported CUDA call names are deduplicated in the summary count."""
        stats = {
            "unsupported_calls": [("cudaDup", "a.cu"), ("cudaDup", "b.cu")],
            "kernel_launches": ["hipLaunchKernelGGL(k1)"],
        }
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            compute_stats(stats)
        output = buf.getvalue()
        self.assertIn("Total number of unsupported CUDA function calls: 1", output)
        self.assertIn("cudaDup", output)
        self.assertIn("Total number of replaced kernel launches: 1", output)

    def test_missing_key_raises(self):
        """Stats dict missing required keys raises KeyError."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            with self.assertRaises(KeyError):
                compute_stats({"kernel_launches": []})
            with self.assertRaises(KeyError):
                compute_stats({"unsupported_calls": []})

    def test_missing_arg_raises(self):
        """Calling compute_stats without stats argument raises TypeError."""
        with self.assertRaises(TypeError):
            compute_stats()


if __name__ == "__main__":
    run_tests()
