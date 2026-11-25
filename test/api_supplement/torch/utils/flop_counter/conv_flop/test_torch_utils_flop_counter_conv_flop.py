# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.flop_counter.conv_flop 接口功能正确性
API 名称：torch.utils.flop_counter.conv_flop
API 签名：torch.utils.flop_counter.conv_flop(x_shape, w_shape, _bias, ..., transposed, *args, out_val=...)

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况                                               |
|------------------|--------------------------------------------|--------------------------------------------------|
| 空/非空          | 零 batch 形状返回 0 与非空卷积形状         | 已覆盖（zero_batch_returns_zero、conv2d_basic）        |
| 枚举选项         | transposed=False 与 transposed=True        | 已覆盖（conv2d_basic、transposed_conv2d）              |
| 参数类型         | 形状列表与 CPU/目标设备 Tensor 输入        | 已覆盖（conv2d_basic、tensor_inputs）                  |
| 传参与不传参     | 可选默认参数省略与显式传入                 | 未覆盖（位置参数必填且 out_val 需显式传入）            |
| 等价类/边界值    | 1D/2D 卷积、转置卷积与零 batch 边界        | 已覆盖（conv1d、transposed_conv2d、zero_batch_returns_zero） |
| 正常传参场景     | 校验正向 FLOPs 与 conv_flop_count 一致     | 已覆盖（conv2d_basic、conv1d、transposed_conv2d、tensor_inputs） |
| 异常传参场景     | 缺少必填位置参数触发 TypeError             | 已覆盖（missing_args_raises）                          |
| 混合设备类型     | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖（仅读取形状元数据计算整数 FLOPs，无设备计算）   |

未覆盖项及原因：
- 传参与不传参：位置参数必填且 out_val 为 FLOPs 计算所需，无独立省略默认分支。
- 混合设备类型：FLOPs 公式仅提取 Tensor.shape 元信息，不涉及跨设备 Tensor 运算。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.utils.flop_counter import conv_flop, conv_flop_count

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
class TestConvFlop(TestCase):
    """Test cases for torch.utils.flop_counter.conv_flop."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_conv2d_basic(self):
        """Computes exact 2D convolution FLOPs matching conv_flop_count."""
        x_shape = [2, 3, 8, 8]
        w_shape = [4, 3, 3, 3]
        out_shape = [2, 4, 6, 6]
        flops = conv_flop(
            x_shape, w_shape, None, [1, 1], [0, 0], [1, 1], False, out_val=out_shape
        )
        expected = conv_flop_count(x_shape, w_shape, out_shape, transposed=False)
        self.assertIsInstance(flops, int)
        self.assertEqual(flops, expected)
        self.assertEqual(flops, 2 * 2 * 4 * 6 * 6 * 3 * 3 * 3)

    def test_conv1d(self):
        """Computes exact 1D convolution FLOPs matching conv_flop_count."""
        x_shape = [2, 3, 12]
        w_shape = [5, 3, 4]
        out_shape = [2, 5, 9]
        flops = conv_flop(
            x_shape, w_shape, None, [1], [0], [1], False, out_val=out_shape
        )
        expected = conv_flop_count(x_shape, w_shape, out_shape, transposed=False)
        self.assertEqual(flops, expected)

    def test_transposed_conv2d(self):
        """Computes exact transposed 2D convolution FLOPs with transposed=True."""
        x_shape = [2, 3, 6, 6]
        w_shape = [3, 4, 3, 3]
        out_shape = [2, 4, 8, 8]
        flops = conv_flop(
            x_shape, w_shape, None, [1, 1], [0, 0], [1, 1], True, out_val=out_shape
        )
        expected = conv_flop_count(x_shape, w_shape, out_shape, transposed=True)
        self.assertEqual(flops, expected)

    def test_zero_batch_returns_zero(self):
        """Returns 0 FLOPs when input batch dimension is zero."""
        flops = conv_flop(
            [0, 3, 8, 8], [4, 3, 3, 3], None, [1, 1], [0, 0], [1, 1], False,
            out_val=[0, 4, 6, 6],
        )
        self.assertEqual(flops, 0)

    def test_tensor_inputs(self):
        """Accepts Tensor inputs on CPU and target device to compute conv2d FLOPs."""
        expected = conv_flop_count([2, 3, 8, 8], [4, 3, 3, 3], [2, 4, 6, 6], transposed=False)
        for dev in (torch.device("cpu"), self.device):
            x = torch.empty(2, 3, 8, 8, device=dev)
            w = torch.empty(4, 3, 3, 3, device=dev)
            out = torch.empty(2, 4, 6, 6, device=dev)
            flops = conv_flop(x, w, None, [1, 1], [0, 0], [1, 1], False, out_val=out)
            self.assertEqual(flops, expected)

    def test_missing_args_raises(self):
        """Calling conv_flop with missing required arguments raises TypeError."""
        with self.assertRaises(TypeError):
            conv_flop()
        with self.assertRaises(TypeError):
            conv_flop([2, 3, 8, 8], [4, 3, 3, 3])


if __name__ == "__main__":
    run_tests()
