# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.q_per_channel_zero_points 接口功能正确性
API 名称：torch.q_per_channel_zero_points
API 签名：torch.q_per_channel_zero_points(input: Tensor) -> Tensor

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 传入非空逐通道量化 Tensor | 已覆盖 |
| 枚举选项 | axis 为 0/1 两种逐通道量化 | 已覆盖 |
| 参数类型 | Tensor 入参 | 已覆盖 |
| 传参与不传参 | input 为必填参数 | 已覆盖 |
| 等价类/边界值 | 返回量化时写入的逐通道 zero_point | 已覆盖 |
| 正常传参场景 | 返回整型 zero_point 张量且数值与构造一致 | 已覆盖 |
| 异常传参场景 | 逐张量量化/非量化输入分别抛错 | 已覆盖 |
| 混合设备类型 | 量化元数据读取，无混合 Tensor 输入 | 未覆盖（不适用） |

未覆盖项及原因：
- 本 API 只读取量化元数据、不接收多设备 Tensor，混合设备维度不适用。
- 量化张量仅在 CPU 上构造，读取 zero_point 的数值语义与设备无关。
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


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class Testq_per_channel_zero_points(TestCase):
    """Test cases for torch.q_per_channel_zero_points (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.base = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.scales = torch.tensor([0.1, 0.2], dtype=torch.float64)
        self.zero_points = torch.tensor([5, 6], dtype=torch.int64)
        self.per_tensor = torch.quantize_per_tensor(
            torch.tensor([-1.0, 0.0, 1.0, 2.0]), 0.1, 10, torch.quint8)

    def test_axis_zero_zero_points(self):
        """Quantizing along axis 0 reports the per-channel zero points for that axis."""
        q = torch.quantize_per_channel(self.base, self.scales, self.zero_points, 0, torch.quint8)
        out = torch.q_per_channel_zero_points(q)
        torch.testing.assert_close(out, self.zero_points, rtol=0, atol=0)

    def test_axis_one_zero_points(self):
        """Quantizing along axis 1 reports the same zero point vector."""
        q = torch.quantize_per_channel(self.base, self.scales, self.zero_points, 1, torch.quint8)
        out = torch.q_per_channel_zero_points(q)
        self.assertEqual(out.shape, (2,))
        torch.testing.assert_close(out, self.zero_points, rtol=0, atol=0)

    def test_returns_int_tensor(self):
        """The zero points are returned as an integer Tensor."""
        q = torch.quantize_per_channel(self.base, self.scales, self.zero_points, 0, torch.quint8)
        out = torch.q_per_channel_zero_points(q)
        self.assertIsInstance(out, torch.Tensor)
        self.assertFalse(out.is_floating_point())

    def test_per_tensor_raises(self):
        """A per-tensor quantized tensor is rejected."""
        with self.assertRaises(RuntimeError):
            torch.q_per_channel_zero_points(self.per_tensor)

    def test_non_quantized_raises(self):
        """A non-quantized tensor has no quantizer and is rejected."""
        with self.assertRaises(NotImplementedError):
            torch.q_per_channel_zero_points(torch.randn(3))


if __name__ == "__main__":
    run_tests()
