# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.hipify.hipify_python.add_dim3 接口功能正确性
API 名称：torch.utils.hipify.hipify_python.add_dim3
API 签名：torch.utils.hipify.hipify_python.add_dim3(kernel_string, cuda_kernel)

覆盖维度表：
| 覆盖维度      | 说明                               | 覆盖情况                                               |
|---------------|------------------------------------|--------------------------------------------------------|
| 空/非空       | 空输入与非空核启动串               | 未覆盖（空串不含 <<<...>>> 启动语法，不属有效调用输入） |
| 枚举选项      | 枚举型参数选项                     | 未覆盖（参数均为字符串，无枚举选项）                   |
| 参数类型      | 合法 str 与非法 None 类型          | 已覆盖（basic_wrap、invalid_type_raises）              |
| 传参与不传参  | 完整传参与缺少必需参数             | 已覆盖（basic_wrap、missing_args_raises）              |
| 等价类/边界值 | 四参数、嵌套括号、空白与已有 dim3  | 已覆盖（four_launch_args、multiline_and_whitespace）   |
| 正常传参场景  | 包裹 grid/block 为 dim3(...)       | 已覆盖（basic_wrap、four_launch_args、nested_parens_in_args） |
| 异常传参场景  | 缺参 TypeError 与非 str 类型异常   | 已覆盖（missing_args_raises、invalid_type_raises）     |
| 混合设备类型  | 多 Tensor 输入时 NPU/CPU 混合      | 未覆盖（纯字符串源码转换工具，无 Tensor 输入）         |

未覆盖项及原因：
- 空/非空：add_dim3 假设输入为已匹配的 CUDA <<<grid, block>>> 启动串，空字符串不满足接口前置条件。
- 枚举选项：参数 kernel_string 与 cuda_kernel 均为字符串，无枚举选项。
- 混合设备类型：接口为纯源码字符串处理工具，无 Tensor 输入，不涉及设备迁移。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.utils.hipify.hipify_python import add_dim3

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
class TestHipifyAddDim3(TestCase):
    """Test cases for torch.utils.hipify.hipify_python.add_dim3."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_basic_wrap(self):
        """Scalar grid and block args are wrapped with dim3(...)."""
        kernel_str = "<<<grid, block>>>"
        cuda_kernel = "my_kernel<<<grid, block>>>("
        result = add_dim3(kernel_str, cuda_kernel)
        self.assertEqual(result, "my_kernel<<<dim3(grid), dim3(block)>>>(")

    def test_four_launch_args(self):
        """Third and fourth launch args (shared mem, stream) remain unwrapped."""
        kernel_str = "<<<grid, block, smem, stream>>>"
        cuda_kernel = "my_kernel<<<grid, block, smem, stream>>>("
        result = add_dim3(kernel_str, cuda_kernel)
        self.assertEqual(result, "my_kernel<<<dim3(grid), dim3(block), smem, stream>>>(")

    def test_nested_parens_in_args(self):
        """Commas inside nested parentheses do not split grid or block args."""
        kernel_str = "<<<dim(1, 2), blk(3, 4), 0>>>"
        cuda_kernel = "launch<<<dim(1, 2), blk(3, 4), 0>>>("
        result = add_dim3(kernel_str, cuda_kernel)
        self.assertEqual(result, "launch<<<dim3(dim(1, 2)), dim3(blk(3, 4)), 0>>>(")

    def test_multiline_and_whitespace(self):
        """Leading/trailing spaces around grid and block args are preserved outside dim3."""
        kernel_str = "<<< grid , block >>>"
        cuda_kernel = "k<<< grid , block >>>("
        result = add_dim3(kernel_str, cuda_kernel)
        self.assertIn("dim3(grid)", result)
        self.assertIn("dim3(block)", result)

    def test_existing_dim3_wrapped_expression(self):
        """Function wraps the first two argument expressions consistently."""
        kernel_str = "<<<dim3(x), dim3(y)>>>"
        cuda_kernel = "k<<<dim3(x), dim3(y)>>>("
        result = add_dim3(kernel_str, cuda_kernel)
        self.assertEqual(result, "k<<<dim3(dim3(x)), dim3(dim3(y))>>>(")

    def test_missing_args_raises(self):
        """Calling add_dim3 without required arguments raises TypeError."""
        with self.assertRaises(TypeError):
            add_dim3()
        with self.assertRaises(TypeError):
            add_dim3("<<<g, b>>>")

    def test_invalid_type_raises(self):
        """Passing non-string kernel_string raises AttributeError."""
        with self.assertRaises(AttributeError):
            add_dim3(None, "k<<<g, b>>>(")


if __name__ == "__main__":
    run_tests()
