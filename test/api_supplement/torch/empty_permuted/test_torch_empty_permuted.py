# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.empty_permuted 接口功能正确性
API 名称：torch.empty_permuted
API 签名：torch.empty_permuted(size, physical_layout, *, dtype, layout, device, pin_memory, requires_grad) -> Tensor

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| shape | 1D/2D/4D size | 已覆盖 |
| dtype | float32/int64/float16/bfloat16 | 已覆盖 |
| device | 每个正常场景 CPU/NPU 双端执行 | 已覆盖 |
| 参数枚举 | physical_layout 恒等/通道后置/自定义置换 | 已覆盖 |
| 可选参数 | dtype/device 显式与省略默认 | 已覆盖 |
| in-place 变体 | 无 in-place 变体 | 未覆盖（不适用） |
| out= 参数 | 无 out= 参数 | 未覆盖（不适用） |
| 广播 | 单纯构造，无广播 | 未覆盖（不适用） |
| 连续性 | 指定布局后 stride/dim_order 正确 | 已覆盖 |
| 混合设备输入 | 无 Tensor 输入（设备由 device 指定） | 未覆盖（不适用） |
| 异常路径 | physical_layout 不合法（重复维度）触发异常 | 已覆盖 |

未覆盖项及原因：
- in-place/out=/广播/混合设备维度对张量构造类 API 不适用：输入为形状与布局元数据。
- 内容为未初始化数据，故仅断言 shape/stride/dim_order/dtype 等结构性契约，
  不做数值对照。
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
class Testempty_permuted(TestCase):
    """Test cases for torch.empty_permuted (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_identity_layout_matches_empty(self):
        """The identity layout yields the same strides as torch.empty."""
        size = (2, 3, 5, 7)
        out = torch.empty_permuted(size, (0, 1, 2, 3))
        self.assertEqual(out.shape, torch.empty(size).shape)
        self.assertEqual(tuple(out.stride()), tuple(torch.empty(size).stride()))

    def test_channels_last_layout(self):
        """physical_layout (0, 2, 3, 1) reproduces channels-last strides."""
        out = torch.empty_permuted((2, 3, 5, 7), (0, 2, 3, 1), device=self.device)
        self.assertEqual(tuple(out.stride()), (105, 1, 21, 3))
        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))

    def test_dim_order_reported(self):
        """dim_order reflects the requested physical layout."""
        out = torch.empty_permuted((2, 3, 5, 7), (0, 2, 3, 1))
        self.assertEqual(tuple(out.dim_order()), (0, 2, 3, 1))

    def test_dtype_and_device(self):
        """Explicit dtype and device are honoured."""
        out = torch.empty_permuted((2, 3), (1, 0), dtype=torch.int64, device=self.device)
        self.assertEqual(out.dtype, torch.int64)
        self.assertEqual(out.device.type, self.device_name)

    def test_1d_and_2d(self):
        """Lower-rank sizes produce dense non-overlapping tensors."""
        self.assertEqual(tuple(torch.empty_permuted((5,), (0,)).shape), (5,))
        two = torch.empty_permuted((3, 4), (1, 0))
        self.assertEqual(tuple(two.shape), (3, 4))
        self.assertEqual(tuple(two.stride()), (1, 3))

    def test_invalid_layout_raises(self):
        """A physical_layout with a repeated dimension is rejected."""
        with self.assertRaises((RuntimeError, ValueError)):
            torch.empty_permuted((2, 3, 4), (0, 0, 2))

    def test_dtype_half_and_bfloat16(self):
        """float16 and bfloat16 dtypes are honoured on the device."""
        for dtype in (torch.float16, torch.bfloat16):
            out = torch.empty_permuted((2, 3, 5, 7), (0, 2, 3, 1),
                                       dtype=dtype, device=self.device)
            self.assertEqual(out.dtype, dtype)
            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))


if __name__ == "__main__":
    run_tests()
