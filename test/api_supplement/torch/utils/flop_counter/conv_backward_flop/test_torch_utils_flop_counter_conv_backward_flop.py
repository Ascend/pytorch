# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.flop_counter.conv_backward_flop 接口功能正确性
API 名称：torch.utils.flop_counter.conv_backward_flop
API 签名：conv_backward_flop(grad_out_shape, x_shape, w_shape, ..., transposed, ..., output_mask, out_val=...)

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况                                               |
|------------------|--------------------------------------------|--------------------------------------------------|
| 空/非空          | 全 False 掩码零计算量与有效梯度掩码        | 已覆盖（output_mask_variants）                         |
| 枚举选项         | output_mask 四组合与 transposed=True/False | 已覆盖（output_mask_variants、transposed_true）        |
| 参数类型         | 形状列表、布尔掩码列表与 out_val 元组      | 已覆盖（output_mask_variants、conv1d_backward）        |
| 传参与不传参     | 可选默认参数省略与显式传入                 | 未覆盖（位置参数均为必填，out_val 需显式传入）         |
| 等价类/边界值    | 全假掩码返回 0、1D/2D 与转置卷积反向       | 已覆盖（output_mask_variants、transposed_true、conv1d_backward） |
| 正常传参场景     | 校验反向 FLOPs 与正向 conv_flop_count 倍数 | 已覆盖（output_mask_variants、transposed_true、conv1d_backward） |
| 异常传参场景     | 缺少必填位置参数触发 TypeError             | 已覆盖（missing_args_raises）                          |
| 混合设备类型     | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖（纯形状元信息 FLOPs 公式，无设备间 Tensor 运算） |

未覆盖项及原因：
- 传参与不传参：位置参数均为必填且 out_val 为计算所必需，无独立省略默认分支。
- 混合设备类型：仅基于形状列表计算整数 FLOPs，不涉及跨设备 Tensor 运算。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.utils.flop_counter import conv_backward_flop, conv_flop_count

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

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
class TestConvBackwardFlop(TestCase):
    """Test cases for torch.utils.flop_counter.conv_backward_flop."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def _call_conv_bwd(self, grad_out, x_shape, w_shape, transposed, output_mask):
        return conv_backward_flop(
            grad_out, x_shape, w_shape, None, [1, 1], [0, 0], [1, 1],
            transposed, [0, 0], 1, output_mask, out_val=(x_shape, w_shape, None),
        )

    def test_output_mask_variants(self):
        """Scales backward conv2d FLOPs according to output_mask gradient flags."""
        x_shape = [2, 3, 8, 8]
        w_shape = [4, 3, 3, 3]
        grad_out = [2, 4, 6, 6]
        fwd = conv_flop_count(x_shape, w_shape, grad_out, transposed=False)
        both = self._call_conv_bwd(grad_out, x_shape, w_shape, False, [True, True, False])
        inp_only = self._call_conv_bwd(grad_out, x_shape, w_shape, False, [True, False, False])
        wt_only = self._call_conv_bwd(grad_out, x_shape, w_shape, False, [False, True, False])
        none_mask = self._call_conv_bwd(grad_out, x_shape, w_shape, False, [False, False, False])
        self.assertEqual(both, 2 * fwd)
        self.assertEqual(inp_only, fwd)
        self.assertEqual(wt_only, fwd)
        self.assertEqual(none_mask, 0)

    def test_transposed_true(self):
        """Computes backward FLOPs for transposed conv2d matching conv_flop_count."""
        x_shape = [2, 3, 6, 6]
        w_shape = [3, 4, 3, 3]
        grad_out = [2, 4, 8, 8]
        fwd = conv_flop_count(x_shape, w_shape, grad_out, transposed=True)
        both = self._call_conv_bwd(grad_out, x_shape, w_shape, True, [True, True, False])
        inp_only = self._call_conv_bwd(grad_out, x_shape, w_shape, True, [True, False, False])
        wt_only = self._call_conv_bwd(grad_out, x_shape, w_shape, True, [False, True, False])
        self.assertEqual(both, 2 * fwd)
        self.assertEqual(inp_only, fwd)
        self.assertEqual(wt_only, fwd)

    def test_conv1d_backward(self):
        """Computes backward FLOPs for 1D convolution matching 2x forward count."""
        x_shape = [2, 3, 10]
        w_shape = [5, 3, 3]
        grad_out = [2, 5, 8]
        fwd = conv_flop_count(x_shape, w_shape, grad_out, transposed=False)
        both = conv_backward_flop(
            grad_out, x_shape, w_shape, None, [1], [0], [1],
            False, [0], 1, [True, True, False], out_val=(x_shape, w_shape, None),
        )
        self.assertEqual(both, 2 * fwd)

    def test_missing_args_raises(self):
        """Calling conv_backward_flop with missing arguments raises TypeError."""
        with self.assertRaises(TypeError):
            conv_backward_flop()
        with self.assertRaises(TypeError):
            conv_backward_flop([2, 4, 6, 6], [2, 3, 8, 8], [4, 3, 3, 3])


if __name__ == "__main__":
    run_tests()
