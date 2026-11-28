# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.wishart.Wishart.scale_tril 接口功能正确性
API 名称：torch.distributions.wishart.Wishart.scale_tril
API 签名：scale_tril(self) -> Tensor（lazy_property）

覆盖维度表：
| 覆盖维度     | 说明                                   | 覆盖情况                              |
|--------------|----------------------------------------|---------------------------------------|
| 空/非空      | 标量批次与批量协方差矩阵形态           | 已覆盖(batched、cholesky_of_cov)    |
| 枚举选项     | covariance/scale_tril 两种构造形态     | 已覆盖（cholesky_of_cov、passthrough）|
| 参数类型     | float64 矩阵张量                       | 已覆盖（各用例统一 float64）          |
| 传参与不传参 | 无入参 lazy_property                   | 已覆盖(各用例均为无参读取)          |
| 等价类/边界  | 2x2 最小维度的 SPD 矩阵                | 已覆盖（cholesky_of_cov）             |
| 正常传参     | 与 cholesky 闭式解及 L@L^T=V 逐元素对照| 已覆盖（cholesky_of_cov、passthrough、|
|              |                                        | reproduces_precision）                |
| 异常传参     | 无异常路径（lazy 求值不触发校验）      | 不适用                                |
| 混合设备类型 | 契约在 CPU 上即可验证，无多设备语义    | 不适用                             |

未覆盖项及原因：
- 未覆盖：scale_tril 的梯度传播。lazy_property 展开为 expand 与
  cholesky 原语，梯度契约由 autograd 原语承担，不在本契约范围。
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
class TestWishartScaleTril(TestCase):
    """Test cases for Wishart.scale_tril (CPU contracts)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_scale_tril_from_covariance_is_cholesky(self):
        """covariance construction yields the Cholesky factor of V."""
        cov = torch.tensor([[2.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
        w = Wishart(torch.tensor(3.0, dtype=torch.float64),
                    covariance_matrix=cov)
        torch.testing.assert_close(w.scale_tril, torch.linalg.cholesky(cov))
        torch.testing.assert_close(
            w.scale_tril @ w.scale_tril.transpose(-2, -1), cov)

    def test_scale_tril_param_passthrough(self):
        """scale_tril construction returns the given factor unchanged."""
        cov = torch.tensor([[2.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
        chol = torch.linalg.cholesky(cov)
        w = Wishart(torch.tensor(3.0, dtype=torch.float64), scale_tril=chol)
        torch.testing.assert_close(w.scale_tril, chol)
        torch.testing.assert_close(w.covariance_matrix, cov)

    def test_scale_tril_is_lower_triangular_with_positive_diag(self):
        """The factor is lower triangular with a positive diagonal."""
        cov = torch.tensor([[2.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
        w = Wishart(torch.tensor(3.0, dtype=torch.float64),
                    covariance_matrix=cov)
        chol = w.scale_tril
        torch.testing.assert_close(chol.tril(), chol)
        self.assertTrue(bool((chol.diagonal(dim1=-2, dim2=-1) > 0).all()))

    def test_scale_tril_reproduces_precision(self):
        """precision computed from scale_tril equals inv(covariance)."""
        cov = torch.tensor([[2.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
        w = Wishart(torch.tensor(3.0, dtype=torch.float64),
                    covariance_matrix=cov)
        identity = torch.eye(2, dtype=torch.float64)
        expected = torch.cholesky_solve(identity, w.scale_tril)
        torch.testing.assert_close(w.precision_matrix, expected)
        torch.testing.assert_close(w.precision_matrix, torch.linalg.inv(cov))

    def test_scale_tril_batched(self):
        """Batched covariance yields per-batch Cholesky factors."""
        cov = torch.stack([torch.eye(2, dtype=torch.float64),
                           2.0 * torch.eye(2, dtype=torch.float64)])
        w = Wishart(torch.tensor(3.0, dtype=torch.float64),
                    covariance_matrix=cov)
        chol = w.scale_tril
        self.assertEqual(tuple(chol.shape), (2, 2, 2))
        torch.testing.assert_close(chol @ chol.transpose(-2, -1), cov)
        torch.testing.assert_close(chol, torch.linalg.cholesky(cov))


if __name__ == "__main__":
    run_tests()
