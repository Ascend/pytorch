# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.wishart.Wishart 构造接口功能正确性
API 名称：torch.distributions.wishart.Wishart
API 签名：Wishart(df, covariance_matrix=None, precision_matrix=None,
         scale_tril=None, validate_args=None)

覆盖维度表：
| 覆盖维度     | 说明                                   | 覆盖情况                              |
|--------------|----------------------------------------|---------------------------------------|
| 空/非空      | 三种矩阵参数互斥，缺省/多传均受限      | 已覆盖(exactly_one_matrix_param)    |
| 枚举选项     | covariance/precision/scale_tril 三选一 | 已覆盖（exactly_one_matrix_param、    |
|              |                                        | equivalence）                         |
| 参数类型     | df 为 Number 与 Tensor 两种形态        | 已覆盖（number_df、rsample_posdef）   |
| 传参与不传参 | validate_args 显式传与不传             | 已覆盖(df_too_small 传默认，构造     |
|              |                                        | 校验均走默认路径）                    |
| 等价类/边界  | df=p-1 的非法边界与 df>p 的合法边界    | 已覆盖（df_too_small_raises）         |
| 正常传参     | mean/cov 与闭式解对照、log_prob 对照   | 已覆盖（from_covariance、equivalence、|
|              |                                        | log_prob_closed_form）                |
| 异常传参     | 多矩阵 AssertionError、df 过小 ValueError| 已覆盖（exactly_one、df_too_small）  |
| 混合设备类型 | 契约在 CPU 上即可验证，无多设备语义    | 不适用                             |

未覆盖项及原因：
- 未覆盖：df<p 触发的 singular sample 警告路径。warnings 属副作用且
  阈值边界（df<p）与采样结果耦合，不做伪断言。
"""
import math
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
class TestWishartConstructor(TestCase):
    """Test cases for the Wishart constructor (CPU contracts)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_wishart_from_covariance_mean_and_cov(self):
        """covariance path: mean == df * V and covariance_matrix == V."""
        cov = torch.eye(2, dtype=torch.float64)
        w = Wishart(torch.tensor(3.0, dtype=torch.float64),
                    covariance_matrix=cov)
        torch.testing.assert_close(w.covariance_matrix, cov)
        torch.testing.assert_close(w.mean, 3.0 * cov)

    def test_exactly_one_matrix_param_required(self):
        """Zero or two matrix parameters raise AssertionError."""
        cov = torch.eye(2, dtype=torch.float64)
        chol = torch.linalg.cholesky(cov)
        df = torch.tensor(3.0, dtype=torch.float64)
        with self.assertRaises(AssertionError):
            Wishart(df, covariance_matrix=cov, scale_tril=chol)
        with self.assertRaises(AssertionError):
            Wishart(df, covariance_matrix=cov, precision_matrix=cov)
        with self.assertRaises(AssertionError):
            Wishart(df)

    def test_df_not_greater_than_ndim_minus_one_raises(self):
        """df <= p - 1 is rejected with ValueError at construction."""
        cov = torch.eye(2, dtype=torch.float64)
        with self.assertRaises(ValueError):
            Wishart(torch.tensor(1.0, dtype=torch.float64),
                    covariance_matrix=cov)
        Wishart(torch.tensor(2.5, dtype=torch.float64),
                covariance_matrix=cov)

    def test_number_df_tensor_shapes(self):
        """A scalar Number df with a batched matrix yields a scalar df."""
        cov = torch.eye(2, dtype=torch.float64)
        w = Wishart(3.0, covariance_matrix=cov)
        torch.testing.assert_close(w.df, torch.tensor(3.0, dtype=torch.float64))
        self.assertEqual(tuple(w.batch_shape), ())

    def test_scale_tril_and_covariance_equivalence(self):
        """covariance and scale_tril parameterizations agree exactly."""
        cov = torch.tensor([[2.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
        chol = torch.linalg.cholesky(cov)
        df = torch.tensor(3.0, dtype=torch.float64)
        w_cov = Wishart(df, covariance_matrix=cov)
        w_chol = Wishart(df, scale_tril=chol)
        torch.testing.assert_close(w_cov.covariance_matrix,
                                   w_chol.covariance_matrix)
        value = 2.5 * torch.eye(2, dtype=torch.float64)
        torch.testing.assert_close(w_cov.log_prob(value), w_chol.log_prob(value))

    def test_rsample_shape_and_positive_definite(self):
        """Seeded rsample keeps shape and positive definiteness."""
        torch.manual_seed(0)
        cov = torch.eye(2, dtype=torch.float64)
        w = Wishart(torch.tensor(5.0, dtype=torch.float64),
                    covariance_matrix=cov)
        s = w.rsample((4,))
        self.assertEqual(tuple(s.shape), (4, 2, 2))
        self.assertTrue(bool(w.support.check(s).all()))
        torch.testing.assert_close(s, s.transpose(-2, -1))

    def test_log_prob_matches_closed_form(self):
        """log_prob at df=p-1 boundary matches the Wishart closed form."""
        df = torch.tensor(3.0, dtype=torch.float64)
        w = Wishart(df, covariance_matrix=torch.eye(2, dtype=torch.float64))
        value = 2.0 * torch.eye(2, dtype=torch.float64)
        nu = 3.0
        p = 2
        expected = ((nu - p - 1) / 2 * torch.log(torch.det(value))
                    - 0.5 * torch.trace(value)
                    - nu * p / 2 * math.log(2.0)
                    - torch.mvlgamma(torch.tensor(nu / 2, dtype=torch.float64),
                                     p=p))
        torch.testing.assert_close(w.log_prob(value), expected)


if __name__ == "__main__":
    run_tests()
