# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.flop_counter.conv_flop_count 接口功能正确性
API 名称：torch.utils.flop_counter.conv_flop_count
API 签名：conv_flop_count(x_shape, w_shape, out_shape, transposed=False) -> int

覆盖维度表：
| 覆盖维度      | 说明                             | 覆盖情况                                                  |
|---------------|----------------------------------|-----------------------------------------------------------|
| 空/非空       | 零维形状与常规非空形状           | 已覆盖（zero_dim_returns_zero、conv2d_formula）           |
| 枚举选项      | transposed=False/True 布尔选项   | 已覆盖（conv2d_formula、transposed_formula）              |
| 参数类型      | list[int] 形状输入与返回 int     | 已覆盖（conv2d_formula、conv1d_and_conv3d）               |
| 传参与不传参  | transposed 默认值与显式传参      | 已覆盖（conv2d_formula、transposed_formula）              |
| 等价类/边界值 | 1D/2D/3D 卷积及 0 维度边界       | 已覆盖（conv1d_and_conv3d、zero_dim_returns_zero）        |
| 正常传参场景  | 标准卷积与转置卷积 FLOP 公式计算 | 已覆盖（conv2d_formula、conv1d_and_conv3d、transposed_formula） |
| 异常传参场景  | 缺少必需参数抛出 TypeError       | 已覆盖（missing_args_raises）                             |
| 混合设备类型  | 多 Tensor 输入时 NPU/CPU 混合    | 未覆盖（纯形状整数列表计算，无 Tensor 输入）              |

未覆盖项及原因：
- 混合设备类型：接口仅接收整数形状列表计算理论 FLOPs，无 Tensor 输入，不涉及设备迁移或混合设备场景。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.utils.flop_counter import conv_flop_count

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
class TestConvFlopCount(TestCase):
    """Test cases for torch.utils.flop_counter.conv_flop_count."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_conv2d_formula(self):
        """Standard 2D conv matches 2 * N * C_out * H_out * W_out * C_in * K_h * K_w."""
        x_shape = [2, 3, 8, 8]
        w_shape = [4, 3, 3, 3]
        out_shape = [2, 4, 6, 6]
        expected = 2 * 2 * 4 * 6 * 6 * 3 * 3 * 3
        flops_default = conv_flop_count(x_shape, w_shape, out_shape)
        flops_explicit = conv_flop_count(x_shape, w_shape, out_shape, transposed=False)
        self.assertIsInstance(flops_default, int)
        self.assertEqual(flops_default, expected)
        self.assertEqual(flops_explicit, expected)

    def test_conv1d_and_conv3d(self):
        """Verify exact FLOP formula for 1D and 3D convolutions."""
        flops_1d = conv_flop_count([2, 3, 10], [5, 3, 4], [2, 5, 7])
        self.assertEqual(flops_1d, 2 * 2 * 5 * 7 * 3 * 4)

        flops_3d = conv_flop_count([2, 3, 6, 6, 6], [4, 3, 3, 3, 3], [2, 4, 4, 4, 4])
        self.assertEqual(flops_3d, 2 * 2 * 4 * (4 * 4 * 4) * 3 * (3 * 3 * 3))

    def test_transposed_formula(self):
        """Transposed conv uses input spatial shape x_shape[2:] instead of out_shape[2:]."""
        x_shape = [2, 3, 5, 7]
        w_shape = [3, 4, 3, 3]
        out_shape = [2, 4, 7, 9]
        flops = conv_flop_count(x_shape, w_shape, out_shape, transposed=True)
        expected = 2 * 2 * 3 * 4 * (5 * 7) * (3 * 3)
        self.assertEqual(flops, expected)
        # Confirm transposed=True differs from transposed=False when spatial dims differ.
        self.assertNotEqual(flops, conv_flop_count(x_shape, w_shape, out_shape, transposed=False))

    def test_zero_dim_returns_zero(self):
        """Zero batch or spatial dimension returns 0 FLOPs."""
        self.assertEqual(conv_flop_count([0, 3, 8, 8], [4, 3, 3, 3], [0, 4, 6, 6]), 0)
        self.assertEqual(conv_flop_count([2, 3, 8, 8], [4, 3, 3, 3], [2, 4, 0, 6]), 0)
        self.assertEqual(conv_flop_count([2, 3, 0, 8], [3, 4, 3, 3], [2, 4, 6, 6], transposed=True), 0)

    def test_missing_args_raises(self):
        """Calling conv_flop_count without required arguments raises TypeError."""
        with self.assertRaises(TypeError):
            conv_flop_count()
        with self.assertRaises(TypeError):
            conv_flop_count([2, 3, 8, 8], [4, 3, 3, 3])


if __name__ == "__main__":
    run_tests()
