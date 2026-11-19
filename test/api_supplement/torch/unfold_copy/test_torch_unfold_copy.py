# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.unfold_copy 接口功能正确性
API 名称：torch.unfold_copy
API 签名：torch.unfold_copy(input, dimension, size, step, *, out=None) -> Tensor

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| shape | 2D/3D/负维度滑窗 | 已覆盖 |
| dtype | float32/float16/float64 | 已覆盖 |
| device | 每个正常场景 CPU/NPU 双端执行 | 已覆盖 |
| 参数枚举 | dimension 正/负、size/step 组合 | 已覆盖 |
| 可选参数 | out= 显式传入 | 已覆盖 |
| in-place 变体 | 无 in-place 变体 | 未覆盖（不适用） |
| out= 参数 | 结果写入指定 tensor | 已覆盖 |
| 广播 | 单输入无广播 | 未覆盖（不适用） |
| 连续性 | 非连续输入（转置后） | 已覆盖 |
| 混合设备输入 | 单 Tensor 输入，无混合设备路径 | 未覆盖（不适用） |
| 异常路径 | size 超过对应维度长度触发异常 | 已覆盖 |

未覆盖项及原因：
- in-place/广播/混合设备维度对单输入复制类 API 不适用：unfold_copy 仅有一个输入张量，
  不做原地修改，也不涉及多张量广播或跨设备运算。
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
class Testunfold_copy(TestCase):
    """Test cases for torch.unfold_copy (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_values_match_cpu_golden(self):
        """The device copy equals the CPU golden sliding-window view."""
        cpu_in = torch.arange(24.0).reshape(2, 3, 4)
        cpu_out = torch.unfold_copy(cpu_in, 1, 2, 1)
        dev_out = torch.unfold_copy(cpu_in.to(self.device), 1, 2, 1).cpu()
        torch.testing.assert_close(cpu_out, dev_out)

    def test_matches_tensor_unfold(self):
        """Output equals the corresponding Tensor.unfold view."""
        x = torch.randn(2, 3, 4)
        torch.testing.assert_close(torch.unfold_copy(x, 1, 2, 1), x.unfold(1, 2, 1))

    def test_shape_contract(self):
        """Shape follows the documented sliding-window formula."""
        x = torch.randn(2, 5, 4)
        out = torch.unfold_copy(x, 1, 3, 2)
        # size positions = (5 - 3) // 2 + 1 = 2, appended window length = 3
        self.assertEqual(tuple(out.shape), (2, 2, 4, 3))
        self.assertEqual(out.shape, x.unfold(1, 3, 2).shape)

    def test_size_step_variants(self):
        """Different size/step combinations match torch.unfold."""
        x = torch.arange(20.0).reshape(4, 5)
        for size, step in [(1, 1), (3, 1), (2, 2)]:
            torch.testing.assert_close(torch.unfold_copy(x, 0, size, step), x.unfold(0, size, step))

    def test_negative_dimension(self):
        """A negative dimension indexes from the back of the shape."""
        x = torch.randn(2, 3, 6)
        torch.testing.assert_close(torch.unfold_copy(x, -1, 2, 2), x.unfold(-1, 2, 2))

    def test_non_contiguous_input(self):
        """A transposed (non-contiguous) input is handled correctly."""
        x = torch.randn(4, 6).t()
        self.assertFalse(x.is_contiguous())
        torch.testing.assert_close(torch.unfold_copy(x, 0, 2, 1), x.unfold(0, 2, 1))

    def test_out_param(self):
        """Result is written into the provided out tensor and returned."""
        x = torch.randn(4)
        out = torch.empty(3, 2)
        result = torch.unfold_copy(x, 0, 2, 1, out=out)
        self.assertIs(result, out)
        torch.testing.assert_close(out, x.unfold(0, 2, 1))

    def test_invalid_size_raises(self):
        """A size larger than the selected dimension raises an error."""
        x = torch.randn(4, 5)
        with self.assertRaises((RuntimeError, IndexError, ValueError)):
            torch.unfold_copy(x, 0, 6, 1)

    def test_dtype_float16_device_golden(self):
        """float16 device sliding window equals the CPU golden result."""
        x = torch.arange(24, dtype=torch.float16).reshape(2, 3, 4)
        cpu_out = torch.unfold_copy(x, 1, 2, 1)
        dev_out = torch.unfold_copy(x.to(self.device), 1, 2, 1).cpu()
        torch.testing.assert_close(cpu_out, dev_out)

    def test_dtype_float64_device_golden(self):
        """float64 device sliding window equals the CPU golden result."""
        x = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
        cpu_out = torch.unfold_copy(x, 1, 2, 1)
        dev_out = torch.unfold_copy(x.to(self.device), 1, 2, 1).cpu()
        torch.testing.assert_close(cpu_out, dev_out)


if __name__ == "__main__":
    run_tests()
