# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.utils.vec_to_tril_matrix 接口功能正确性
API 名称：torch.distributions.utils.vec_to_tril_matrix
API 签名：vec_to_tril_matrix(vec: Tensor, diag: int = 0) -> Tensor

覆盖维度表：
| 覆盖维度     | 说明                                   | 覆盖情况                              |
|--------------|----------------------------------------|---------------------------------------|
| 空/非空      | 单向量与批量向量输入                   | 已覆盖（single_element_vec、          |
|              |                                        | batched_roundtrip）                   |
| 枚举选项     | diag 取 -1/0/1 等离散取值              | 已覆盖（positive_diag、negative_diag）|
| 参数类型     | float32 向量/批量向量张量              | 已覆盖（batched_roundtrip）           |
| 传参与不传参 | 不传参默认 diag=0；显式传 -1/1         | 已覆盖（diag_zero_basic、             |
|              |                                        | positive_diag、negative_diag）        |
| 等价类/边界  | 长度 1 的最小可表达向量                | 已覆盖（single_element_vec）          |
| 正常传参     | 与 tril_matrix_to_vec 互逆、形状还原   | 已覆盖（positive_diag、               |
|              |                                        | batched_roundtrip）                   |
| 异常传参     | 向量长度不可表达为三角元素数时 ValueError| 已覆盖(invalid_size_raises)        |
| 混合设备类型 | 契约在 CPU 上即可验证，无多设备语义    | 不适用                             |

未覆盖项及原因：
- 未覆盖：空向量（长度 0）输入。长度 0 时求根结果非正整数，属于
  invalid_size 用例的同族行为，无独立断言价值。
"""
import os
import unittest

import torch

from torch.distributions.utils import tril_matrix_to_vec, vec_to_tril_matrix

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
    # No torch_npu (e.g. a plain CUDA/CPU box). Fall back to the stdlib
    # TestCase. This file does not compare floating point tensors, so the
    # fallback is safe here.
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

# transfer_to_npu rebinds torch.cuda.* and torch.Event to the NPU
# implementations, so the test body uses generic entry points unmodified
# on NPU and CUDA. Keep AFTER the testcase import above.
if DEVICE_NAME == 'npu':
    import torch_npu.contrib.transfer_to_npu  # noqa: F401


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestVecToTrilMatrix(TestCase):
    """Test cases for utils.vec_to_tril_matrix (row-order rebuild)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_diag_zero_basic(self):
        """Three elements rebuild a 2x2 lower-triangular matrix."""
        vec = torch.tensor([1.0, 2.0, 3.0])
        expected = torch.tensor([[1.0, 0.0], [2.0, 3.0]])
        torch.testing.assert_close(vec_to_tril_matrix(vec), expected)

    def test_positive_diag(self):
        """diag=1 rebuilds a 3x3 matrix including the superdiagonal."""
        vec = torch.tensor([1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        mat = vec_to_tril_matrix(vec, diag=1)
        self.assertEqual(tuple(mat.shape), (3, 3))
        torch.testing.assert_close(tril_matrix_to_vec(mat, diag=1), vec)

    def test_negative_diag(self):
        """diag=-1 rebuilds a strict lower-triangular 3x3 matrix."""
        vec = torch.tensor([4.0, 7.0, 8.0])
        expected = torch.tensor(
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [7.0, 8.0, 0.0]])
        torch.testing.assert_close(vec_to_tril_matrix(vec, diag=-1), expected)

    def test_batched_roundtrip(self):
        """Batched vec -> mat -> vec round-trips exactly."""
        torch.manual_seed(0)
        mat = torch.randn(2, 3, 3)
        vec = tril_matrix_to_vec(mat)
        mat2 = vec_to_tril_matrix(vec)
        self.assertEqual(tuple(mat2.shape), (2, 3, 3))
        torch.testing.assert_close(tril_matrix_to_vec(mat2), vec)
        torch.testing.assert_close(mat2, torch.tril(mat))

    def test_invalid_size_raises(self):
        """Lengths not expressible as tril element counts raise."""
        with self.assertRaises(ValueError):
            vec_to_tril_matrix(torch.randn(5))
        with self.assertRaises(ValueError):
            vec_to_tril_matrix(torch.randn(2))

    def test_single_element_vec(self):
        """A length-1 vector is the minimal valid 1x1 matrix."""
        vec = torch.tensor([7.0])
        mat = vec_to_tril_matrix(vec)
        self.assertEqual(tuple(mat.shape), (1, 1))
        torch.testing.assert_close(mat, torch.tensor([[7.0]]))


if __name__ == "__main__":
    run_tests()
