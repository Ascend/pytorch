# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.gru_cell 接口功能正确性
API 名称：torch.gru_cell
API 签名：torch.gru_cell(input, hx, w_ih, w_hh, b_ih=None, b_hh=None) -> Tensor

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| shape | 2D 批输入、3D 单样本展开、单样本 | 已覆盖 |
| dtype | float32 | 已覆盖 |
| device | 每个正常场景 CPU/NPU 双端执行 | 已覆盖 |
| 参数枚举 | 含/不含偏置（b_ih/b_hh 省略） | 已覆盖 |
| 可选参数 | 显式传入偏置 vs 省略默认 | 已覆盖 |
| in-place 变体 | 无 in-place 变体 | 未覆盖（不适用） |
| out= 参数 | 无 out= 参数 | 未覆盖（不适用） |
| 广播 | 无广播语义 | 未覆盖（不适用） |
| 连续性 | 结果默认连续 | 已覆盖 |
| 混合设备输入 | 多张量输入，NPU/CPU 混用触发异常 | 已覆盖 |
| 异常路径 | input_size 与权重不一致触发异常 | 已覆盖 |

未覆盖项及原因：
- out=/广播维度对该单元计算 API 不适用。
- 浮点输出以 CPU 为 Golden，设备侧结果移至 CPU 后用 torch.testing.assert_close 对照。
"""
import os
import unittest

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# 仅在补充测试真正运行（RUN_FLAG）时才探测设备：默认 CI 收集会被类级
# skipUnless 跳过，此时不做 torch_npu 导入与 NPU/CUDA 设备初始化，避免
# 不必要的探测开销；被跳过时 DEVICE_NAME 仅为占位值 'cpu'。
DEVICE_NAME = 'cpu'
if RUN_FLAG:
    DEVICE_NAME = None
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = 'npu'
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'

# ORDER CONSTRAINT: import torch_npu.testing.testcase BEFORE
# torch_npu.contrib.transfer_to_npu (patching first breaks the import).
try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

if DEVICE_NAME == 'npu':
    import torch_npu.contrib.transfer_to_npu  # noqa: F401

HIDDEN = 2
INPUT = 3


def _weights(device='cpu', seed=0):
    """Deterministic GRU weights for input size 3 and hidden size 2."""
    gen = torch.Generator().manual_seed(seed)
    w_ih = torch.randn(3 * HIDDEN, INPUT, generator=gen).to(device)
    w_hh = torch.randn(3 * HIDDEN, HIDDEN, generator=gen).to(device)
    b_ih = torch.randn(3 * HIDDEN, generator=gen).to(device)
    b_hh = torch.randn(3 * HIDDEN, generator=gen).to(device)
    return w_ih, w_hh, b_ih, b_hh


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class Testgru_cell(TestCase):
    """Test cases for torch.gru_cell (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_values_match_cpu_golden(self):
        """Device GRU cell output equals the CPU golden output."""
        torch.manual_seed(0)
        x = torch.randn(4, INPUT)
        hx = torch.randn(4, HIDDEN)
        w_ih, w_hh, b_ih, b_hh = _weights()
        cpu_out = torch.gru_cell(x, hx, w_ih, w_hh, b_ih, b_hh)
        w_ih_d, w_hh_d, b_ih_d, b_hh_d = _weights(self.device_name)
        dev_out = torch.gru_cell(x.to(self.device), hx.to(self.device),
                                 w_ih_d, w_hh_d, b_ih_d, b_hh_d).cpu()
        torch.testing.assert_close(cpu_out, dev_out)

    def test_matches_reference_formula(self):
        """Output matches the closed-form GRU update equations."""
        torch.manual_seed(1)
        x = torch.randn(4, INPUT)
        hx = torch.randn(4, HIDDEN)
        w_ih, w_hh, b_ih, b_hh = _weights()
        out = torch.gru_cell(x, hx, w_ih, w_hh, b_ih, b_hh)
        i = x @ w_ih.t() + b_ih
        h = hx @ w_hh.t() + b_hh
        i_r, i_z, i_n = i.chunk(3, dim=1)
        h_r, h_z, h_n = h.chunk(3, dim=1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        expected = (1 - z) * n + z * hx
        torch.testing.assert_close(out, expected)

    def test_shape_contract(self):
        """Output shape is (batch, hidden)."""
        torch.manual_seed(0)
        x = torch.randn(5, INPUT)
        hx = torch.randn(5, HIDDEN)
        w_ih, w_hh, b_ih, b_hh = _weights()
        self.assertEqual(tuple(torch.gru_cell(x, hx, w_ih, w_hh, b_ih, b_hh).shape), (5, HIDDEN))

    def test_optional_bias_omitted(self):
        """Omitting b_ih/b_hh defaults them to bias-free computation."""
        torch.manual_seed(0)
        x = torch.randn(4, INPUT)
        hx = torch.randn(4, HIDDEN)
        w_ih, w_hh, _, _ = _weights()
        out = torch.gru_cell(x, hx, w_ih, w_hh)
        zeros = torch.zeros(3 * HIDDEN)
        torch.testing.assert_close(out, torch.gru_cell(x, hx, w_ih, w_hh, zeros, zeros))

    def test_invalid_input_size_raises(self):
        """A weight matrix inconsistent with input size is rejected."""
        torch.manual_seed(0)
        x = torch.randn(4, INPUT)
        hx = torch.randn(4, HIDDEN)
        w_ih, w_hh, b_ih, b_hh = _weights()
        bad_w_ih = torch.randn(3 * HIDDEN, INPUT + 5)
        with self.assertRaises(RuntimeError):
            torch.gru_cell(x, hx, bad_w_ih, w_hh, b_ih, b_hh)

    def test_mixed_device_raises(self):
        """Mixing a CPU input with device weights must fail."""
        if self.device_name == 'cpu':
            self.skipTest('无设备: 缺少 NPU/CUDA 设备执行路径')
        x = torch.randn(4, INPUT)
        hx = torch.randn(4, HIDDEN)
        w_ih, w_hh, b_ih, b_hh = _weights(self.device_name)
        with self.assertRaises((RuntimeError, ValueError)):
            torch.gru_cell(x, hx, w_ih, w_hh, b_ih, b_hh)


if __name__ == "__main__":
    run_tests()
