# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.hipify.hipify_python.hip_header_magic 接口功能正确性
API 名称：torch.utils.hipify.hipify_python.hip_header_magic
API 签名：torch.utils.hipify.hipify_python.hip_header_magic(input_string)

覆盖维度表：
| 覆盖维度      | 说明                                 | 覆盖情况                                              |
|---------------|--------------------------------------|-------------------------------------------------------|
| 空/非空       | 空字符串与非空设备/主机代码文本      | 已覆盖（empty_string、adds_header_for_device_code）   |
| 枚举选项      | 枚举型参数选项                       | 未覆盖（仅接收源码字符串 input_string，无枚举选项）   |
| 参数类型      | str 源码输入与 str 返回值            | 已覆盖（adds_header_for_device_code、empty_string）   |
| 传参与不传参  | 传入 input_string 与缺参调用         | 已覆盖（adds_header_for_device_code、missing_arg_raises） |
| 等价类/边界值 | 已有 hip 头文件、纯主机代码与无关头  | 已覆盖（existing_header_not_duplicated、host_only_unchanged） |
| 正常传参场景  | 设备代码自动补齐 hip_runtime.h       | 已覆盖（adds_header_for_device_code、existing_header_not_duplicated） |
| 异常传参场景  | 缺参抛出 TypeError                   | 已覆盖（missing_arg_raises）                          |
| 混合设备类型  | 多 Tensor 输入时 NPU/CPU 混合        | 未覆盖（纯源码文本处理函数，无 Tensor 输入）          |

未覆盖项及原因：
- 枚举选项：接口仅接收单个源码字符串参数 input_string，无枚举类参数。
- 混合设备类型：接口仅检查源码关键字并插入头文件声明，无 Tensor 输入，不涉及设备迁移。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.utils.hipify.hipify_python import hip_header_magic

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
class TestHipifyHipHeaderMagic(TestCase):
    """Test cases for torch.utils.hipify.hipify_python.hip_header_magic."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_adds_header_for_device_code(self):
        """Prepends hip_runtime.h when device markers like __global__ are present."""
        for snippet in (
            "__global__ void k() {}\n",
            "__shared__ float buf[16];\n",
            "hipLaunchKernelGGL(k, 1, 1, 0, 0);\n",
            "void f() { __syncthreads(); }\n",
        ):
            result = hip_header_magic(snippet)
            self.assertEqual(result, '#include "hip/hip_runtime.h"\n' + snippet)

    def test_existing_header_not_duplicated(self):
        """Does not prepend header if hip_runtime.h or hip_runtime_api.h is included."""
        with_quotes = '#include "hip/hip_runtime.h"\n__global__ void k() {}\n'
        self.assertEqual(hip_header_magic(with_quotes), with_quotes)

        with_angles = "#include <hip/hip_runtime_api.h>\n__global__ void k() {}\n"
        self.assertEqual(hip_header_magic(with_angles), with_angles)

    def test_host_only_unchanged(self):
        """Plain host C++ code without device logic is returned unchanged."""
        src = "#include <vector>\nint add(int a, int b) { return a + b; }\n"
        self.assertEqual(hip_header_magic(src), src)

    def test_empty_string(self):
        """Empty string returns empty string."""
        self.assertEqual(hip_header_magic(""), "")

    def test_unrelated_include_unchanged(self):
        """Unrelated includes without device markers remain untouched."""
        src = "#include <iostream>\n#include <cmath>\nint main() { return 0; }\n"
        self.assertEqual(hip_header_magic(src), src)

    def test_missing_arg_raises(self):
        """Calling hip_header_magic without input_string raises TypeError."""
        with self.assertRaises(TypeError):
            hip_header_magic()


if __name__ == "__main__":
    run_tests()
