# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.wishart.Wishart.precision_matrix 接口功能正确性
API 名称：torch.distributions.wishart.Wishart.precision_matrix
API 签名：precision_matrix(self) -> Tensor（lazy_property）

覆盖维度表：
| 覆盖维度     | 说明                                   | 覆盖情况                              |
|--------------|----------------------------------------|---------------------------------------|
| 空/非空      | 标量批次与批量 df 参数形态             | 已覆盖（batched_df、inverse_of_cov）  |
| 枚举选项     | covariance/precision 两种构造形态      | 已覆盖（inverse_of_cov、from_param）  |
| 参数类型     | float64 矩阵张量与 df 张量             | 已覆盖（各用例统一 float64）          |
| 传参与不传参 | 无入参 lazy_property                   | 已覆盖(各用例均为无参读取)          |
| 等价类/边界  | 2x2 最小维度的 SPD 矩阵                | 已覆盖（inverse_of_cov）              |
| 正常传参     | 与 linalg.inv 闭式解及 P@V=I 逐元素对照| 已覆盖（inverse_of_cov、times_cov、   |
|              |                                        | from_param_roundtrip）                |
| 异常传参     | 无异常路径（lazy 求值不触发校验）      | 不适用                                |
| 混合设备类型 | 契约在 CPU 上即可验证，无多设备语义    | 不适用                             |

未覆盖项及原因：
- 未覆盖：precision_matrix 的梯度传播。lazy_property 求值路径含
  cholesky_solve，梯度契约由 autograd 原语承担，不在本契约范围。

注意：df 用 0 维标量张量构造（torch 2.15 契约 batch_shape =
     broadcast_shapes(param.shape[:-2], df.shape)），torch.tensor([3.0]) 会
     引入多余的 (1,) 批维使 lazy 结果膨胀为 (1,2,2)，与 (2,2) 期望失配；
     批量 df 用例（batched_df）的 (2,) 形状为有意为之。
"""
import os
import unittest

import torch

from torch.distributions.wishart import Wishart

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
class TestWishartPrecisionMatrix(TestCase):
    """Test cases for Wishart.precision_matrix (CPU contracts)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_precision_is_inverse_of_covariance(self):
        """precision_matrix equals the matrix inverse of covariance."""
        cov = torch.tensor([[2.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
        w = Wishart(torch.tensor(3.0, dtype=torch.float64),
                    covariance_matrix=cov)
        torch.testing.assert_close(w.covariance_matrix, cov)
        torch.testing.assert_close(w.precision_matrix, torch.linalg.inv(cov))

    def test_precision_times_covariance_is_identity(self):
        """P @ V recovers the identity matrix."""
        cov = torch.tensor([[2.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
        w = Wishart(torch.tensor(3.0, dtype=torch.float64),
                    covariance_matrix=cov)
        product = w.precision_matrix @ w.covariance_matrix
        torch.testing.assert_close(product, torch.eye(2, dtype=torch.float64))

    def test_precision_from_precision_param_roundtrip(self):
        """precision construction round-trips; covariance is its inverse."""
        prec = torch.tensor([[2.0, 0.5], [0.5, 1.5]], dtype=torch.float64)
        w = Wishart(torch.tensor(3.0, dtype=torch.float64),
                    precision_matrix=prec)
        torch.testing.assert_close(w.precision_matrix, prec)
        torch.testing.assert_close(w.covariance_matrix, torch.linalg.inv(prec))

    def test_precision_symmetric_positive_definite(self):
        """The result is symmetric with strictly positive eigenvalues."""
        cov = torch.tensor([[2.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
        w = Wishart(torch.tensor(3.0, dtype=torch.float64),
                    covariance_matrix=cov)
        prec = w.precision_matrix
        torch.testing.assert_close(prec, prec.transpose(-2, -1))
        self.assertTrue(bool((torch.linalg.eigvalsh(prec) > 0).all()))

    def test_precision_batched_df(self):
        """Batched df expands the result per batch with the same inverse."""
        df = torch.tensor([3.0, 4.0], dtype=torch.float64)
        cov = torch.tensor([[2.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
        w = Wishart(df, covariance_matrix=cov)
        self.assertEqual(tuple(w.batch_shape), (2,))
        prec = w.precision_matrix
        self.assertEqual(tuple(prec.shape), (2, 2, 2))
        self.assertTrue(bool((torch.linalg.eigvalsh(prec) > 0).all()))
        torch.testing.assert_close(
            prec, torch.linalg.inv(cov).expand(2, 2, 2))


if __name__ == "__main__":
    run_tests()
