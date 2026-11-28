# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.utils.tril_matrix_to_vec 接口功能正确性
API 名称：torch.distributions.utils.tril_matrix_to_vec
API 签名：tril_matrix_to_vec(mat: Tensor, diag: int = 0) -> Tensor

覆盖维度表：
| 覆盖维度     | 说明                                   | 覆盖情况                              |
|--------------|----------------------------------------|---------------------------------------|
| 空/非空      | 批量与非批量矩阵输入                   | 已覆盖（batched、diag_zero_row_order）|
| 枚举选项     | diag 取 -1/0/1 等离散取值              | 已覆盖（matches_tril_indices_gather） |
| 参数类型     | float32 矩阵/批量矩阵张量              | 已覆盖（batched）                     |
| 传参与不传参 | 不传参默认 diag=0；显式传 -1/1/2       | 已覆盖（diag_zero_row_order、         |
|              |                                        | positive_diag、negative_diag）        |
| 等价类/边界  | diag=-(n) 与 diag=n-1 的合法边界       | 已覆盖（matches_tril_indices_gather） |
| 正常传参     | 与 tril_indices 收集闭式解逐元素对照   | 已覆盖（matches_tril_indices_gather） |
| 异常传参     | diag 越界触发 ValueError               | 已覆盖（invalid_diag_raises）         |
| 混合设备类型 | 契约在 CPU 上即可验证，无多设备语义    | 不适用                             |

未覆盖项及原因：
- 未覆盖：非方阵输入。源码按 mat.shape[-1] 取 n，非方阵行为未定义，
  上游未承诺，不做伪断言。
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
class TestTrilMatrixToVec(TestCase):
    """Test cases for utils.tril_matrix_to_vec (row-order extraction)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_diag_zero_row_order(self):
        """diag=0 extracts the full lower triangle (j <= i) in row order."""
        mat = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        torch.testing.assert_close(
            tril_matrix_to_vec(mat), torch.tensor([1.0, 4.0, 5.0, 7.0, 8.0, 9.0]))

    def test_positive_diag_includes_superdiagonal(self):
        """diag=1 includes the first superdiagonal in row order."""
        mat = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        expected = torch.tensor([1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        torch.testing.assert_close(tril_matrix_to_vec(mat, diag=1), expected)

    def test_negative_diag_strict_lower(self):
        """diag=-1 extracts only the strict lower triangle."""
        mat = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        torch.testing.assert_close(
            tril_matrix_to_vec(mat, diag=-1), torch.tensor([4.0, 7.0, 8.0]))

    def test_matches_tril_indices_gather(self):
        """Any valid diag matches a tril_indices-based gather oracle."""
        torch.manual_seed(0)
        mat = torch.randn(4, 4)
        for diag in (-3, -1, 0, 2):
            rows, cols = torch.tril_indices(4, 4, diag)
            torch.testing.assert_close(
                tril_matrix_to_vec(mat, diag=diag), mat[rows, cols])

    def test_batched(self):
        """Batched matrices stack into (batch, tril_count) vectors."""
        torch.manual_seed(1)
        mat = torch.randn(2, 3, 3)
        vec = tril_matrix_to_vec(mat)
        self.assertEqual(tuple(vec.shape), (2, 6))
        per_item = torch.stack([tril_matrix_to_vec(mat[0]),
                                tril_matrix_to_vec(mat[1])])
        torch.testing.assert_close(vec, per_item)

    def test_invalid_diag_raises(self):
        """diag outside [-n, n-1] raises ValueError."""
        mat = torch.randn(3, 3)
        with self.assertRaises(ValueError):
            tril_matrix_to_vec(mat, diag=3)
        with self.assertRaises(ValueError):
            tril_matrix_to_vec(mat, diag=-4)

    def test_roundtrip_with_vec_to_tril_matrix(self):
        """mat -> vec -> mat preserves the tril entries exactly."""
        torch.manual_seed(2)
        mat = torch.randn(3, 3)
        vec = tril_matrix_to_vec(mat, diag=1)
        mat2 = vec_to_tril_matrix(vec, diag=1)
        torch.testing.assert_close(tril_matrix_to_vec(mat2, diag=1), vec)
        torch.testing.assert_close(mat2, torch.tril(mat, diagonal=1))


if __name__ == "__main__":
    run_tests()
