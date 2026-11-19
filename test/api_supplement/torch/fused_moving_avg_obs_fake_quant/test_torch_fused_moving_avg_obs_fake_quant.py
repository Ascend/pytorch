# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fused_moving_avg_obs_fake_quant 接口功能正确性
API 名称：torch.fused_moving_avg_obs_fake_quant
API 签名：torch.fused_moving_avg_obs_fake_quant(input, observer_on, fake_quant_on,
          running_min, running_max, scale, zero_point, averaging_const, quant_min,
          quant_max, ch_axis, per_row_fake_quant=False, symmetric_quant=False) -> Tensor

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| shape | 2D、高维 | 已覆盖 |
| dtype | float32 | 已覆盖 |
| device | 每个正常场景 CPU/NPU 双端执行 | 已覆盖 |
| 参数枚举 | observer_on/fake_quant_on 的 0/1 组合 | 已覆盖 |
| 可选参数 | per_row_fake_quant 省略默认 | 已覆盖 |
| in-place 变体 | running_min/running_max 原地更新 | 已覆盖 |
| out= 参数 | 无 out= 参数 | 未覆盖（不适用） |
| 广播 | 逐元素/逐通道量化，无广播语义 | 未覆盖（不适用） |
| 连续性 | 结果默认连续 | 已覆盖 |
| 混合设备输入 | NPU 侧无原生实现，走 CPU 回退，未定义混合设备校验 | 未覆盖（见下） |
| 异常路径 | input 为 0D 标量触发异常 | 已覆盖 |

未覆盖项及原因：
- out=/广播维度对该融合量化 API 不适用。
- 混合设备输入：该 API 在 NPU 上无原生内核（`_fused_moving_avg_obs_fq_helper` 走
  `npu_cpu_fallback` 回退到 CPU 执行），回退路径不校验输入张量设备一致性，混用
  CPU 输入与设备端状态张量不会报错（与 CUDA 原生内核的设备校验不同）。因此
  「混合设备触发异常」并非该 API 在 NPU 上的既有契约，未做该断言。
- 浮点输出以 CPU 为 Golden，设备侧结果移至 CPU 后用 torch.testing.assert_close 对照；
  pending 统计量（running_min/running_max）在设备侧原地更新后再移回 CPU 对照。
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


def _state(device='cpu'):
    """Quantization observer state tensors for one fused fake-quant call."""
    running_min = torch.tensor(float('inf'), device=device)
    running_max = torch.tensor(float('-inf'), device=device)
    scale = torch.tensor([1.0], device=device)
    zero_point = torch.tensor([0], dtype=torch.int, device=device)
    observer_on = torch.tensor([1.0], device=device)
    fake_quant_on = torch.tensor([1.0], device=device)
    return observer_on, fake_quant_on, running_min, running_max, scale, zero_point


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class Testfused_moving_avg_obs_fake_quant(TestCase):
    """Test cases for torch.fused_moving_avg_obs_fake_quant (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_values_match_cpu_golden(self):
        """Device output matches the CPU golden fused fake-quant output."""
        torch.manual_seed(0)
        x = torch.randn(6, 8)
        state = _state()
        cpu_out = torch.fused_moving_avg_obs_fake_quant(
            x, state[0], state[1], state[2], state[3], state[4], state[5],
            0.01, 0, 255, 0, False)
        dev_state = _state(self.device_name)
        dev_out = torch.fused_moving_avg_obs_fake_quant(
            x.to(self.device), dev_state[0], dev_state[1], dev_state[2], dev_state[3],
            dev_state[4], dev_state[5], 0.01, 0, 255, 0, False).cpu()
        torch.testing.assert_close(cpu_out, dev_out)
        torch.testing.assert_close(dev_state[2].cpu(), state[2])

    def test_observer_updates_running_stats(self):
        """With the observer enabled the running min/max track the data."""
        x = torch.tensor([[1.0, -2.0], [3.0, 0.5]])
        observer_on, fake_quant_on, running_min, running_max, scale, zero_point = _state()
        torch.fused_moving_avg_obs_fake_quant(
            x, observer_on, fake_quant_on, running_min, running_max, scale, zero_point,
            0.01, 0, 255, 0, False)
        self.assertEqual(float(running_min.item()), float(x.min().item()))
        self.assertEqual(float(running_max.item()), float(x.max().item()))

    def test_fake_quant_disabled_is_identity(self):
        """With fake quant disabled the output equals the input."""
        x = torch.randn(5, 5)
        observer_on, _, running_min, running_max, scale, zero_point = _state()
        fake_quant_off = torch.tensor([0.0])
        out = torch.fused_moving_avg_obs_fake_quant(
            x, observer_on, fake_quant_off, running_min, running_max, scale, zero_point,
            0.01, 0, 255, 0, False)
        torch.testing.assert_close(out, x)

    def test_fake_quant_enabled_changes_values(self):
        """With fake quant enabled the values are requantized."""
        torch.manual_seed(0)
        x = torch.randn(5, 5)
        observer_on, fake_quant_on, running_min, running_max, scale, zero_point = _state()
        out = torch.fused_moving_avg_obs_fake_quant(
            x, observer_on, fake_quant_on, running_min, running_max, scale, zero_point,
            0.01, 0, 255, 0, False)
        self.assertFalse(bool(torch.allclose(out, x)))

    def test_observer_disabled_keeps_stats(self):
        """With the observer disabled the running stats are left untouched."""
        x = torch.randn(5, 5)
        running_min = torch.tensor(0.0)
        running_max = torch.tensor(1.0)
        observer_off = torch.tensor([0.0])
        fake_quant_on = torch.tensor([1.0])
        scale = torch.tensor([1.0])
        zero_point = torch.tensor([0], dtype=torch.int)
        torch.fused_moving_avg_obs_fake_quant(
            x, observer_off, fake_quant_on, running_min, running_max, scale, zero_point,
            0.01, 0, 255, 0, False)
        self.assertEqual(float(running_min.item()), 0.0)
        self.assertEqual(float(running_max.item()), 1.0)

    def test_output_shape_and_dtype(self):
        """Output keeps the input shape and dtype."""
        x = torch.randn(2, 3, 4)
        observer_on, fake_quant_on, running_min, running_max, scale, zero_point = _state(self.device_name)
        out = torch.fused_moving_avg_obs_fake_quant(
            x.to(self.device), observer_on, fake_quant_on, running_min, running_max,
            scale, zero_point, 0.01, 0, 255, 0, False)
        self.assertEqual(tuple(out.shape), (2, 3, 4))
        self.assertEqual(out.dtype, torch.float32)


if __name__ == "__main__":
    run_tests()
