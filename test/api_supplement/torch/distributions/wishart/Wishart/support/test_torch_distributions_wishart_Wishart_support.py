# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.wishart.Wishart.support 接口功能正确性
API 名称：torch.distributions.wishart.Wishart.support
API 签名：support = constraints.positive_definite（类属性）

覆盖维度表：
| 覆盖维度     | 说明                                   | 覆盖情况                              |
|--------------|----------------------------------------|---------------------------------------|
| 空/非空      | 零矩阵（退化）与非退化 SPD 矩阵        | 已覆盖（check_boundary_values）       |
| 枚举选项     | 本接口无枚举入参                       | 不适用                                |
| 参数类型     | float64 矩阵张量与 df 张量             | 已覆盖（check_on_rsampled_matrices）  |
| 传参与不传参 | 无入参类属性；check 显式传矩阵         | 已覆盖(check_boundary_values 等)    |
| 等价类/边界  | 半正定/负定/奇异三类边界矩阵           | 已覆盖(singular_and_indefinite、     |
|              |                                        | check_boundary_values）               |
| 正常传参     | 约束对象同一性与 check 布尔结果断言    | 已覆盖(is_positive_definite)        |
| 异常传参     | check 返回 False 而非抛错，无异常路径  | 不适用                                |
| 混合设备类型 | 契约在 CPU 上即可验证，无多设备语义    | 不适用                             |

未覆盖项及原因：
- 未覆盖：非方阵或 1 维输入的 check 结果。positive_definite 的
  cholesky_ex 校验对非方阵的行为未在源码中稳定承诺，不做伪断言。

注意：df 用 0 维标量张量构造（torch 2.15 契约 batch_shape =
     broadcast_shapes(param.shape[:-2], df.shape)），torch.tensor([3.0]) 会
     引入多余的 (1,) 批维使 rsample 形状变为 (3,1,2,2)，与 (3,2,2) 期望失配。
"""
import os
import unittest

import torch

from torch.distributions import constraints
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


def _make_wishart():
    """Build a scalar-batched float64 Wishart for support checks."""
    return Wishart(torch.tensor(3.0, dtype=torch.float64),
                   covariance_matrix=torch.eye(2, dtype=torch.float64))


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestWishartSupport(TestCase):
    """Test cases for Wishart.support (positive definite constraint)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_support_is_positive_definite(self):
        """support is the shared positive_definite constraint object."""
        w = _make_wishart()
        self.assertIs(w.support, constraints.positive_definite)
        self.assertIs(Wishart.support, constraints.positive_definite)

    def test_support_check_boundary_values(self):
        """SPD matrices pass; zero and negative-definite matrices fail."""
        w = _make_wishart()
        self.assertTrue(bool(
            w.support.check(torch.eye(2, dtype=torch.float64)).all()))
        self.assertFalse(bool(
            w.support.check(torch.zeros(2, 2, dtype=torch.float64)).any()))
        self.assertFalse(bool(
            w.support.check(-torch.eye(2, dtype=torch.float64)).any()))

    def test_support_check_on_rsampled_matrices(self):
        """All seeded rsample outputs lie in the positive-definite support."""
        torch.manual_seed(0)
        w = Wishart(torch.tensor(5.0, dtype=torch.float64),
                    covariance_matrix=torch.eye(2, dtype=torch.float64))
        s = w.rsample((3,))
        self.assertEqual(tuple(s.shape), (3, 2, 2))
        self.assertTrue(bool(w.support.check(s).all()))

    def test_support_check_singular_and_indefinite(self):
        """Singular and indefinite matrices are rejected."""
        w = _make_wishart()
        singular = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
        indefinite = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float64)
        self.assertFalse(bool(w.support.check(singular).any()))
        self.assertFalse(bool(w.support.check(indefinite).any()))


if __name__ == "__main__":
    run_tests()
