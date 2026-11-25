# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.flop_counter.baddbmm_flop 接口功能正确性
API 名称：torch.utils.flop_counter.baddbmm_flop
API 签名：torch.utils.flop_counter.baddbmm_flop(self_shape, a_shape, b_shape, out_shape=None, **kwargs)

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况                                               |
|------------------|--------------------------------------------|--------------------------------------------------|
| 空/非空          | 含零维度形状返回 0 与非空三维形状          | 已覆盖（zero_dim_returns_zero、basic_formula）         |
| 枚举选项         | 接口枚举类参数覆盖                         | 未覆盖（接口无枚举类参数）                             |
| 参数类型         | tuple、list 与 CPU/目标设备 Tensor 输入    | 已覆盖（basic_formula、various_sizes、tensor_inputs）  |
| 传参与不传参     | out_shape 可选参数省略与显式传入           | 未覆盖（out_shape 为占位参数，用例覆盖默认三必填入参） |
| 等价类/边界值    | 单/多 batch 矩形尺寸与各维为 0 边界        | 已覆盖（various_sizes、zero_dim_returns_zero）         |
| 正常传参场景     | 校验 2*B*M*K*N 精确 FLOPs 计算             | 已覆盖（basic_formula、various_sizes、tensor_inputs）  |
| 异常传参场景     | batch/内维不匹配抛 AssertionError 与缺参   | 已覆盖（mismatched_dims_raises、missing_args_raises）  |
| 混合设备类型     | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖（仅读取形状元数据计算整数 FLOPs，无设备计算）   |

未覆盖项及原因：
- 枚举选项：接口无枚举类参数。
- 传参与不传参：out_shape 与 **kwargs 不参与公式计算，当前用例覆盖必填三形状参数路径。
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
from torch.utils.flop_counter import baddbmm_flop

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
class TestBaddbmmFlop(TestCase):
    """Test cases for torch.utils.flop_counter.baddbmm_flop."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_basic_formula(self):
        """Verify exact FLOP formula 2 * B * M * K * N for typical shapes."""
        b, m, k, n = 4, 8, 16, 6
        flops = baddbmm_flop((b, m, n), (b, m, k), (b, k, n))
        self.assertIsInstance(flops, int)
        self.assertEqual(flops, 2 * b * m * k * n)

    def test_various_sizes(self):
        """Verify FLOP count across single-batch and multi-batch rectangular shapes."""
        cases = [
            (1, 1, 1, 1),
            (1, 5, 7, 3),
            (3, 12, 4, 9),
            (8, 2, 32, 16),
        ]
        for b, m, k, n in cases:
            flops = baddbmm_flop([b, m, n], [b, m, k], [b, k, n])
            self.assertEqual(flops, 2 * b * m * k * n)

    def test_zero_dim_returns_zero(self):
        """Zero in batch or matrix dimensions produces 0 FLOPs."""
        self.assertEqual(baddbmm_flop((0, 4, 5), (0, 4, 3), (0, 3, 5)), 0)
        self.assertEqual(baddbmm_flop((2, 0, 5), (2, 0, 3), (2, 3, 5)), 0)
        self.assertEqual(baddbmm_flop((2, 4, 5), (2, 4, 0), (2, 0, 5)), 0)
        self.assertEqual(baddbmm_flop((2, 4, 0), (2, 4, 3), (2, 3, 0)), 0)

    def test_tensor_inputs(self):
        """Wrapped formula accepts Tensor inputs on CPU and target device."""
        b, m, k, n = 2, 3, 4, 5
        expected = 2 * b * m * k * n
        for dev in (torch.device("cpu"), self.device):
            self_t = torch.empty(b, m, n, device=dev)
            a_t = torch.empty(b, m, k, device=dev)
            b_t = torch.empty(b, k, n, device=dev)
            self.assertEqual(baddbmm_flop(self_t, a_t, b_t), expected)

    def test_mismatched_dims_raises(self):
        """Mismatched batch or inner dimensions raise AssertionError."""
        with self.assertRaises(AssertionError):
            baddbmm_flop((2, 4, 5), (2, 4, 3), (3, 3, 5))
        with self.assertRaises(AssertionError):
            baddbmm_flop((2, 4, 5), (2, 4, 3), (2, 7, 5))

    def test_missing_args_raises(self):
        """Calling baddbmm_flop with missing required arguments raises TypeError."""
        with self.assertRaises(TypeError):
            baddbmm_flop()
        with self.assertRaises(TypeError):
            baddbmm_flop((2, 4, 5), (2, 4, 3))


if __name__ == "__main__":
    run_tests()
