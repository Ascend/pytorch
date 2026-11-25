# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.flop_counter.sdpa_flop 接口功能正确性
API 名称：torch.utils.flop_counter.sdpa_flop
API 签名：sdpa_flop(query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs) -> int

覆盖维度表：
| 覆盖维度      | 说明                              | 覆盖情况                                               |
|---------------|-----------------------------------|--------------------------------------------------------|
| 空/非空       | 零序列长度与常规非空形状          | 已覆盖（zero_dim_and_tensor_inputs、basic_formula）    |
| 枚举选项      | is_causal=False/True 布尔选项     | 已覆盖（extra_kwargs_accepted）                        |
| 参数类型      | tuple 形状与 CPU/目标设备 Tensor  | 已覆盖（basic_formula、zero_dim_and_tensor_inputs）    |
| 传参与不传参  | 额外关键字参数省略与显式传入      | 已覆盖（basic_formula、extra_kwargs_accepted）         |
| 等价类/边界值 | 0 序列边界、GQA 与 batch 线性缩放 | 已覆盖（zero_dim_and_tensor_inputs、gqa_and_linear_scaling） |
| 正常传参场景  | 标准 SDPA、GQA 与带 kwargs 场景   | 已覆盖（basic_formula、gqa_and_linear_scaling、extra_kwargs_accepted） |
| 异常传参场景  | 形状不匹配及缺参异常              | 已覆盖（mismatched_shapes_and_missing_args_raises）    |
| 混合设备类型  | 多 Tensor 输入时 NPU/CPU 混合     | 未覆盖（仅提取 Tensor.shape 计算，不涉及跨设备运算）   |

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
from torch.utils.flop_counter import sdpa_flop

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
class TestSdpaFlop(TestCase):
    """Test cases for torch.utils.flop_counter.sdpa_flop."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_basic_formula(self):
        """Verify exact formula 2 * B * H * S_q * S_k * (D_qk + D_v)."""
        b, h, s_q, s_k, d_qk, d_v = 2, 4, 16, 12, 8, 6
        q_shape = (b, h, s_q, d_qk)
        k_shape = (b, h, s_k, d_qk)
        v_shape = (b, h, s_k, d_v)
        flops = sdpa_flop(q_shape, k_shape, v_shape)
        self.assertIsInstance(flops, int)
        self.assertEqual(flops, 2 * b * h * s_q * s_k * (d_qk + d_v))

    def test_gqa_and_linear_scaling(self):
        """Verify GQA (h_q multiple of h_kv) and linear scaling with batch/heads."""
        b, h_q, h_kv, s_q, s_k, d = 2, 8, 2, 10, 14, 16
        flops = sdpa_flop((b, h_q, s_q, d), (b, h_kv, s_k, d), (b, h_kv, s_k, d))
        self.assertEqual(flops, 2 * b * h_q * s_q * s_k * (2 * d))

        double_b = sdpa_flop((2 * b, h_q, s_q, d), (2 * b, h_kv, s_k, d), (2 * b, h_kv, s_k, d))
        self.assertEqual(double_b, 2 * flops)

    def test_extra_kwargs_accepted(self):
        """Extra keyword arguments such as is_causal are accepted without error."""
        shape = (2, 4, 8, 8)
        expected = 2 * 2 * 4 * 8 * 8 * (8 + 8)
        self.assertEqual(sdpa_flop(shape, shape, shape, is_causal=False), expected)
        causal_flops = sdpa_flop(shape, shape, shape, is_causal=True)
        self.assertIsInstance(causal_flops, int)
        self.assertGreater(causal_flops, 0)

    def test_zero_dim_and_tensor_inputs(self):
        """Zero sequence length yields 0; Tensor inputs work on CPU and target device."""
        self.assertEqual(sdpa_flop((2, 4, 0, 8), (2, 4, 8, 8), (2, 4, 8, 8)), 0)
        for dev in (torch.device("cpu"), self.device):
            q = torch.empty(1, 2, 4, 6, device=dev)
            k = torch.empty(1, 2, 5, 6, device=dev)
            v = torch.empty(1, 2, 5, 4, device=dev)
            self.assertEqual(sdpa_flop(q, k, v), 2 * 1 * 2 * 4 * 5 * (6 + 4))

    def test_mismatched_shapes_and_missing_args_raises(self):
        """Incompatible shapes raise AssertionError; missing args raise TypeError."""
        with self.assertRaises(AssertionError):
            sdpa_flop((2, 4, 8, 8), (3, 4, 8, 8), (2, 4, 8, 8))
        with self.assertRaises(AssertionError):
            sdpa_flop((2, 5, 8, 8), (2, 2, 8, 8), (2, 2, 8, 8))
        with self.assertRaises(TypeError):
            sdpa_flop()
        with self.assertRaises(TypeError):
            sdpa_flop((2, 4, 8, 8), (2, 4, 8, 8))


if __name__ == "__main__":
    run_tests()
