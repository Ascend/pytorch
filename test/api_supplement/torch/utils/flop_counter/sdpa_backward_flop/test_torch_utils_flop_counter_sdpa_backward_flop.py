# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.flop_counter.sdpa_backward_flop 接口功能正确性
API 名称：torch.utils.flop_counter.sdpa_backward_flop
API 签名：sdpa_backward_flop(grad_out_shape, query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs)

覆盖维度表：
| 覆盖维度      | 说明                               | 覆盖情况                                               |
|---------------|------------------------------------|--------------------------------------------------------|
| 空/非空       | 零序列长度与常规非空 4D 形状       | 已覆盖（zero_dim_and_tensor_inputs、basic_formula）    |
| 枚举选项      | is_causal 等可选布尔关键字参数     | 已覆盖（linear_scaling_and_extra_kwargs）              |
| 参数类型      | tuple 形状与 Tensor 输入           | 已覆盖（basic_formula、zero_dim_and_tensor_inputs）    |
| 传参与不传参  | 额外关键字参数省略与显式传入       | 已覆盖（basic_formula、linear_scaling_and_extra_kwargs） |
| 等价类/边界值 | 0 序列边界、GQA 与非对称 S/D 维度  | 已覆盖（zero_dim_and_tensor_inputs、gqa_and_asymmetric_dims） |
| 正常传参场景  | 标准 4D、GQA 及线性缩放 FLOP 公式  | 已覆盖（basic_formula、gqa_and_asymmetric_dims）       |
| 异常传参场景  | 形状不匹配及缺参异常               | 已覆盖（mismatched_shapes_and_missing_args_raises）    |
| 混合设备类型  | 多 Tensor 输入时 NPU/CPU 混合      | 未覆盖（仅提取 Tensor.shape 计算，不涉及跨设备运算）   |

未覆盖项及原因：
- 混合设备类型：接口通过 shape_wrapper 仅读取输入 Tensor 的形状信息计算理论 FLOPs，不执行跨设备张量运算。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.utils.flop_counter import sdpa_backward_flop

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
class TestSdpaBackwardFlop(TestCase):
    """Test cases for torch.utils.flop_counter.sdpa_backward_flop."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_basic_formula(self):
        """Verify exact backward FLOP formula for 4D (B, H, S, D) shapes."""
        b, h, s, d = 2, 4, 16, 8
        shape = (b, h, s, d)
        flops = sdpa_backward_flop(shape, shape, shape, shape)
        self.assertIsInstance(flops, int)
        self.assertGreater(flops, 0)
        self.assertEqual(flops, 2 * b * h * s * s * (3 * d + 2 * d))

    def test_gqa_and_asymmetric_dims(self):
        """Verify formula with GQA (h_q > h_kv) and distinct s_q/s_k, d_q/d_v."""
        b, h_q, h_kv, s_q, s_k, d_q, d_v = 2, 8, 2, 12, 20, 16, 10
        grad_out = (b, h_q, s_q, d_v)
        q = (b, h_q, s_q, d_q)
        k = (b, h_kv, s_k, d_q)
        v = (b, h_kv, s_k, d_v)
        flops = sdpa_backward_flop(grad_out, q, k, v)
        expected = 2 * b * h_q * s_q * s_k * (3 * d_q + 2 * d_v)
        self.assertEqual(flops, expected)

    def test_linear_scaling_and_extra_kwargs(self):
        """FLOPs scale linearly with batch/heads and accept extra kwargs."""
        base = sdpa_backward_flop((1, 2, 8, 4), (1, 2, 8, 4), (1, 2, 8, 4), (1, 2, 8, 4))
        double_b = sdpa_backward_flop((2, 2, 8, 4), (2, 2, 8, 4), (2, 2, 8, 4), (2, 2, 8, 4))
        double_h = sdpa_backward_flop((1, 4, 8, 4), (1, 4, 8, 4), (1, 4, 8, 4), (1, 4, 8, 4))
        self.assertEqual(double_b, 2 * base)
        self.assertEqual(double_h, 2 * base)

        with_kw = sdpa_backward_flop(
            (1, 2, 8, 4), (1, 2, 8, 4), (1, 2, 8, 4), (1, 2, 8, 4), is_causal=True
        )
        self.assertIsInstance(with_kw, int)
        self.assertGreater(with_kw, 0)

    def test_zero_dim_and_tensor_inputs(self):
        """Zero sequence length returns 0; Tensor inputs work via shape_wrapper."""
        self.assertEqual(sdpa_backward_flop((2, 4, 0, 8), (2, 4, 0, 8), (2, 4, 8, 8), (2, 4, 8, 8)), 0)
        q = torch.empty(1, 2, 4, 4, device=self.device)
        expected = 2 * 1 * 2 * 4 * 4 * (3 * 4 + 2 * 4)
        self.assertEqual(sdpa_backward_flop(q, q, q, q), expected)

    def test_mismatched_shapes_and_missing_args_raises(self):
        """Mismatched shapes raise AssertionError; missing args raise TypeError."""
        with self.assertRaises(AssertionError):
            sdpa_backward_flop((2, 4, 8, 8), (3, 4, 8, 8), (2, 4, 8, 8), (2, 4, 8, 8))
        with self.assertRaises(AssertionError):
            sdpa_backward_flop((2, 5, 8, 8), (2, 5, 8, 8), (2, 2, 8, 8), (2, 2, 8, 8))
        with self.assertRaises(TypeError):
            sdpa_backward_flop()
        with self.assertRaises(TypeError):
            sdpa_backward_flop((2, 4, 8, 8), (2, 4, 8, 8))


if __name__ == "__main__":
    run_tests()
