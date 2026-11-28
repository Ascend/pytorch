# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.constraint_registry.transform_to 接口功能正确性
API 名称：torch.distributions.constraint_registry.transform_to
API 签名：transform_to = ConstraintRegistry()；__call__(self, constraint) -> Transform
         （源码 constraint_registry.py:124-153，注册表见 162-291）

覆盖维度表：
| 覆盖维度     | 说明                                              | 覆盖情况                                  |
|--------------|---------------------------------------------------|-------------------------------------------|
| 空/非空      | 空/非空指注册表键存在与否（未注册约束报错）       | 已覆盖（unregistered_constraint_raises）  |
| 枚举选项     | real/positive/simplex/interval/greater_than_eq/   | 已覆盖（各 transform 用例逐枚举覆盖）     |
|              | less_than/lower_cholesky/positive_definite        |                                           |
| 参数类型     | 入参为 Constraint 对象（含参数化约束实例）        | 已覆盖（less_than、greater_than_eq 用例） |
| 传参与不传参 | 单参形态，constraint 必传                         | 已覆盖（各用例）                          |
| 等价类/边界  | 边界：softmax 输出和为 1、PD 判定、inv 往返       | 已覆盖（simplex、positive_definite 用例） |
| 正常传参     | 返回 Transform 且输出落在目标约束内               | 已覆盖（各 transform 用例）               |
| 异常传参     | 未注册约束抛 NotImplementedError                  | 已覆盖（unregistered_constraint_raises）  |
| 混合设备类型 | 注册表为纯 Python 对象，无设备迁移语义            | 不适用                                    |

未覆盖项及原因：
- 未覆盖：cat/stack/corr_cholesky 注册项。查表契约与所选枚举一致，避免冗余；
  transform_to 与 biject_to 的差异点（simplex 选型不同）已单独覆盖。

注意：所有浮点结果用 torch.testing.assert_close 对照解析值或约束 check。
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
    # No torch_npu (e.g. a plain CUDA/CPU box). Fall back to the stdlib
    # TestCase. Floating-point comparisons below use the device-agnostic
    # torch.testing.assert_close, so the fallback is safe here.
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
class TestTransformTo(TestCase):
    """Contract tests for torch.distributions.constraint_registry.transform_to."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_real_returns_identity(self):
        """transform_to(real) must return the identity transform singleton."""
        t = torch.distributions.constraint_registry.transform_to(
            torch.distributions.constraints.real
        )
        self.assertIs(t, torch.distributions.transforms.identity_transform)
        x = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float64)
        torch.testing.assert_close(t(x), x)

    def test_positive_returns_exp_transform(self):
        """transform_to(positive) maps the real line onto (0, inf) with a
        log-density adjustment: the returned transform is either a bare
        ExpTransform or a ComposeTransform containing one (2.15 builds
        compose an AffineTransform for the constraint's lower bound)."""
        constraints = torch.distributions.constraints
        transforms = torch.distributions.transforms
        t = torch.distributions.constraint_registry.transform_to(constraints.positive)
        parts = t.parts if isinstance(t, transforms.ComposeTransform) else [t]
        self.assertTrue(any(isinstance(p, transforms.ExpTransform) for p in parts))
        x = torch.tensor([-3.0, 0.0, 2.0], dtype=torch.float64)
        y = t(x)
        torch.testing.assert_close(y, torch.exp(x))
        self.assertEqual((y > 0.0).all().item(), True)
        torch.testing.assert_close(t.inv(y), x)

    def test_simplex_returns_softmax_not_stick_breaking(self):
        """transform_to vs biject_to differ on simplex: softmax vs stick-breaking."""
        constraints = torch.distributions.constraints
        transforms = torch.distributions.transforms
        transform_to = torch.distributions.constraint_registry.transform_to
        biject_to = torch.distributions.constraint_registry.biject_to
        softmax = transform_to(constraints.simplex)
        stick = biject_to(constraints.simplex)
        self.assertIsInstance(softmax, transforms.SoftmaxTransform)
        self.assertIsInstance(stick, transforms.StickBreakingTransform)
        # Module contract: softmax overparameterizes and is NOT bijective,
        # while the stick-breaking bijection is.
        self.assertIs(softmax.bijective, False)
        self.assertIs(stick.bijective, True)
        torch.manual_seed(201)
        x = torch.randn(5, 4, dtype=torch.float64)
        # SoftmaxTransform is shift-invariant (softmax(x+c) == softmax(x)), so
        # its inv recovers x only up to the log-sum-exp constant. Normalize x
        # onto the zero-lse slice first, then inv(softmax(x)) == x exactly.
        x = x - torch.logsumexp(x, dim=-1, keepdim=True)
        y = softmax(x)
        torch.testing.assert_close(y.sum(-1), torch.ones(5, dtype=torch.float64))
        self.assertEqual((y > 0.0).all().item(), True)
        torch.testing.assert_close(softmax.inv(y), x, atol=1e-10, rtol=1e-10)

    def test_unit_interval_returns_sigmoid(self):
        """transform_to(interval(0, 1)) maps the real line into (0, 1)."""
        constraints = torch.distributions.constraints
        transforms = torch.distributions.transforms
        t = torch.distributions.constraint_registry.transform_to(
            constraints.unit_interval
        )
        self.assertIsInstance(t, transforms.SigmoidTransform)
        x = torch.tensor([-5.0, 0.0, 5.0], dtype=torch.float64)
        y = t(x)
        torch.testing.assert_close(y, torch.sigmoid(x))
        self.assertEqual(((y > 0.0) & (y < 1.0)).all().item(), True)

    def test_greater_than_eq_and_less_than_bounds(self):
        """greater_than_eq maps above the bound, less_than maps below it."""
        constraints = torch.distributions.constraints
        transforms = torch.distributions.transforms
        transform_to = torch.distributions.constraint_registry.transform_to
        t_ge = transform_to(constraints.greater_than_eq(1.5))
        self.assertIsInstance(t_ge, transforms.ComposeTransform)
        x_ge = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float64)
        y_ge = t_ge(x_ge)
        self.assertEqual((y_ge >= 1.5).all().item(), True)
        torch.testing.assert_close(y_ge, 1.5 + torch.exp(x_ge))
        t_lt = transform_to(constraints.less_than(1.0))
        self.assertIsInstance(t_lt, transforms.ComposeTransform)
        y_lt = t_lt(torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float64))
        self.assertEqual((y_lt < 1.0).all().item(), True)

    def test_lower_cholesky_transform(self):
        """transform_to(lower_cholesky) produces lower-triangular positive
        diag; inv recovers the encoded lower-triangle part of the input (the
        upper triangle of the unconstrained input is discarded by the
        transform's image, so it is not compared)."""
        constraints = torch.distributions.constraints
        transforms = torch.distributions.transforms
        t = torch.distributions.constraint_registry.transform_to(
            constraints.lower_cholesky
        )
        self.assertIsInstance(t, transforms.LowerCholeskyTransform)
        torch.manual_seed(202)
        x = torch.randn(4, 3, 3, dtype=torch.float64)
        y = t(x)
        self.assertEqual(constraints.lower_cholesky.check(y).all().item(), True)
        torch.testing.assert_close(t.inv(y), x.tril())

    def test_positive_definite_transform(self):
        """transform_to(positive_definite) output must satisfy the PD constraint."""
        constraints = torch.distributions.constraints
        transforms = torch.distributions.transforms
        t = torch.distributions.constraint_registry.transform_to(
            constraints.positive_definite
        )
        self.assertIsInstance(t, transforms.PositiveDefiniteTransform)
        torch.manual_seed(203)
        x = torch.randn(2, 3, 3, dtype=torch.float64)
        y = t(x)
        self.assertEqual(constraints.positive_definite.check(y).all().item(), True)

    def test_unregistered_constraint_raises(self):
        """Constraints without a transform_to entry must raise NotImplementedError."""
        constraints = torch.distributions.constraints
        constraint = constraints.MixtureSameFamilyConstraint(constraints.real)
        with self.assertRaisesRegex(NotImplementedError, "Cannot transform"):
            torch.distributions.constraint_registry.transform_to(constraint)


if __name__ == "__main__":
    run_tests()
