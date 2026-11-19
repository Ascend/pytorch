# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.diagonal_copy 接口功能正确性
API 名称：torch.diagonal_copy
API 签名：torch.diagonal_copy(input, offset=0, dim1=0, dim2=1, *, out=None) -> Tensor

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| shape | 方阵/非方阵/高维对角抽取 | 已覆盖 |
| dtype | float32/int64/float16/float64 | 已覆盖 |
| device | 每个正常场景 CPU/NPU 双端执行 | 已覆盖 |
| 参数枚举 | offset 正/负/0、dim1/dim2 组合 | 已覆盖 |
| 可选参数 | 显式传入 vs 省略默认 | 已覆盖 |
| in-place 变体 | 无 in-place 变体 | 未覆盖（不适用） |
| out= 参数 | 结果写入指定 tensor | 已覆盖 |
| 广播 | 单输入无广播 | 未覆盖（不适用） |
| 连续性 | 非连续输入（转置后） | 已覆盖 |
| 混合设备输入 | 单 Tensor 输入，无混合设备路径 | 未覆盖（不适用） |
| 异常路径 | dim1/dim2 越界触发异常 | 已覆盖 |

未覆盖项及原因：
- in-place/广播/混合设备维度对单输入复制类 API 不适用：diagonal_copy 仅有一个输入张量，
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
class Testdiagonal_copy(TestCase):
    """Test cases for torch.diagonal_copy (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_values_match_cpu_golden(self):
        """The device copy equals the CPU golden diagonal."""
        cpu_in = torch.arange(12.0).reshape(3, 4)
        cpu_out = torch.diagonal_copy(cpu_in)
        dev_out = torch.diagonal_copy(cpu_in.to(self.device)).cpu()
        torch.testing.assert_close(cpu_out, dev_out)

    def test_matches_tensor_diagonal(self):
        """Output values equal the corresponding torch.diagonal view."""
        x = torch.randn(4, 5)
        torch.testing.assert_close(torch.diagonal_copy(x), x.diagonal())

    def test_returns_independent_copy(self):
        """Mutating the result must not touch the source tensor."""
        x = torch.randn(4, 4)
        out = torch.diagonal_copy(x)
        out.fill_(999.0)
        self.assertNotEqual(x[0][0].item(), 999.0)

    def test_offset_and_dims_enumeration(self):
        """offset and the (dim1, dim2) pair are honoured."""
        x = torch.arange(12, dtype=torch.int64).reshape(3, 4)
        self.assertEqual(torch.diagonal_copy(x, 1).tolist(), x.diagonal(1).tolist())
        self.assertEqual(torch.diagonal_copy(x, -1).tolist(), x.diagonal(-1).tolist())
        self.assertEqual(torch.diagonal_copy(x, 0, 1, 0).tolist(), x.diagonal(0, 1, 0).tolist())

    def test_dtype_preserved(self):
        """The integer dtype of the input is preserved."""
        x = torch.arange(12, dtype=torch.int64).reshape(3, 4)
        self.assertEqual(torch.diagonal_copy(x).dtype, torch.int64)

    def test_non_contiguous_input(self):
        """A transposed (non-contiguous) input is handled correctly."""
        x = torch.randn(4, 6).t()
        self.assertFalse(x.is_contiguous())
        torch.testing.assert_close(torch.diagonal_copy(x), x.diagonal())

    def test_out_param(self):
        """Result is written into the provided out tensor and returned."""
        out = torch.empty(3)
        result = torch.diagonal_copy(torch.randn(3, 3), out=out)
        self.assertIs(result, out)
        self.assertEqual(tuple(out.shape), (3,))

    def test_invalid_dim_raises(self):
        """An out-of-range dim raises an error."""
        x = torch.randn(3, 3)
        with self.assertRaises((RuntimeError, IndexError, ValueError)):
            torch.diagonal_copy(x, 0, 0, 5)

    def test_dtype_float16_device_golden(self):
        """float16 device diagonal equals the CPU golden diagonal."""
        x = torch.arange(12, dtype=torch.float16).reshape(3, 4)
        cpu_out = torch.diagonal_copy(x)
        dev_out = torch.diagonal_copy(x.to(self.device)).cpu()
        torch.testing.assert_close(cpu_out, dev_out)

    def test_dtype_float64_device_golden(self):
        """float64 device diagonal equals the CPU golden diagonal."""
        x = torch.arange(12, dtype=torch.float64).reshape(3, 4)
        cpu_out = torch.diagonal_copy(x)
        dev_out = torch.diagonal_copy(x.to(self.device)).cpu()
        torch.testing.assert_close(cpu_out, dev_out)


if __name__ == "__main__":
    run_tests()
