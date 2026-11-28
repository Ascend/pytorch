# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.constraint_registry.biject_to 接口功能正确性
API 名称：torch.distributions.constraint_registry.biject_to
API 签名：biject_to = ConstraintRegistry()；__call__(self, constraint) -> Transform
         （源码 constraint_registry.py:124-153，注册表见 162-291）

覆盖维度表：
| 覆盖维度     | 说明                                              | 覆盖情况                                  |
|--------------|---------------------------------------------------|-------------------------------------------|
| 空/非空      | 空/非空指注册表键存在与否（未注册约束报错）       | 已覆盖（unregistered_constraint_raises）  |
| 枚举选项     | real/positive/greater_than/unit_interval/simplex/ | 已覆盖（各 transform 用例逐枚举覆盖）     |
|              | independent 等已注册约束选项                      |                                           |
| 参数类型     | 入参为 Constraint 对象（含参数化约束实例）        | 已覆盖（greater_than、interval 用例）     |
| 传参与不传参 | 单参形态，constraint 必传                         | 已覆盖（各用例）                          |
| 等价类/边界  | 边界：0/1 处 sigmoid 渐近、单位单纯形求和为 1     | 已覆盖（unit_interval、simplex 用例）     |
| 正常传参     | 返回 bijective Transform 且往返逆一致             | 已覆盖（各 transform 用例）               |
| 异常传参     | 未注册约束抛 NotImplementedError                  | 已覆盖（unregistered_constraint_raises）  |
| 混合设备类型 | 注册表为纯 Python 对象，无设备迁移语义            | 不适用                                    |

未覆盖项及原因：
- 未覆盖：cat/stack/lower_cholesky/corr_cholesky 等复合注册项。它们复用同一
  查表契约（类型 -> Transform），核心行为已被所选枚举覆盖，避免用例冗余。

注意：所有浮点结果（正逆变换、log_abs_det_jacobian）均用
     torch.testing.assert_close 与解析值对照。
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
class TestBijectTo(TestCase):
    """Contract tests for torch.distributions.constraint_registry.biject_to."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_real_returns_identity(self):
        """biject_to(real) must return the identity transform singleton."""
        t = torch.distributions.constraint_registry.biject_to(
            torch.distributions.constraints.real
        )
        self.assertIs(t, torch.distributions.transforms.identity_transform)
        x = torch.tensor([-1.5, 0.0, 2.5], dtype=torch.float64)
        torch.testing.assert_close(t(x), x)
        torch.testing.assert_close(t.inv(t(x)), x)

    def test_positive_returns_exp_transform(self):
        """biject_to(positive) maps the real line bijectively onto (0, inf)
        with J = y. The returned transform is either a bare ExpTransform or
        a ComposeTransform containing one (2.15 builds compose an affine
        shift for the GreaterThanEq lower bound)."""
        constraints = torch.distributions.constraints
        transforms = torch.distributions.transforms
        t = torch.distributions.constraint_registry.biject_to(constraints.positive)
        parts = t.parts if isinstance(t, transforms.ComposeTransform) else [t]
        self.assertTrue(any(isinstance(p, transforms.ExpTransform) for p in parts))
        self.assertIs(t.bijective, True)
        x = torch.tensor([-2.0, 0.0, 1.5], dtype=torch.float64)
        y = t(x)
        torch.testing.assert_close(y, torch.exp(x))
        torch.testing.assert_close(t.inv(y), x)
        torch.testing.assert_close(t.log_abs_det_jacobian(x, y), x)

    def test_greater_than_returns_exp_affine_compose(self):
        """biject_to(greater_than(2.0)) maps real line onto (2, inf) bijectively."""
        constraints = torch.distributions.constraints
        transforms = torch.distributions.transforms
        constraint = constraints.greater_than(2.0)
        t = torch.distributions.constraint_registry.biject_to(constraint)
        self.assertIsInstance(t, transforms.ComposeTransform)
        self.assertIs(t.bijective, True)
        x = torch.tensor([-3.0, 0.0, 4.0], dtype=torch.float64)
        y = t(x)
        # Every unconstrained input must land strictly above the lower bound.
        self.assertEqual((y > 2.0).all().item(), True)
        torch.testing.assert_close(y, 2.0 + torch.exp(x))
        torch.testing.assert_close(t.inv(y), x)

    def test_unit_interval_returns_sigmoid(self):
        """biject_to(interval(0, 1)) must be the bijective SigmoidTransform."""
        constraints = torch.distributions.constraints
        transforms = torch.distributions.transforms
        t = torch.distributions.constraint_registry.biject_to(
            constraints.unit_interval
        )
        self.assertIsInstance(t, transforms.SigmoidTransform)
        self.assertIs(t.bijective, True)
        x = torch.tensor([-4.0, 0.0, 3.0], dtype=torch.float64)
        y = t(x)
        torch.testing.assert_close(y, torch.sigmoid(x))
        # Boundary semantics: outputs stay in the open interval (0, 1).
        self.assertEqual((y > 0.0).all().item(), True)
        self.assertEqual((y < 1.0).all().item(), True)
        torch.testing.assert_close(t.inv(y), x)

    def test_simplex_returns_stick_breaking(self):
        """biject_to(simplex) must be the bijective StickBreakingTransform."""
        constraints = torch.distributions.constraints
        transforms = torch.distributions.transforms
        t = torch.distributions.constraint_registry.biject_to(constraints.simplex)
        self.assertIsInstance(t, transforms.StickBreakingTransform)
        self.assertIs(t.bijective, True)
        torch.manual_seed(101)
        x = torch.randn(7, 3, dtype=torch.float64)
        y = t(x)
        torch.testing.assert_close(y.sum(-1), torch.ones(7, dtype=torch.float64))
        self.assertEqual((y > 0.0).all().item(), True)
        torch.testing.assert_close(t.inv(y), x, atol=1e-10, rtol=1e-10)

    def test_independent_wraps_base_transform(self):
        """biject_to(independent(positive, 1)) yields an IndependentTransform."""
        constraints = torch.distributions.constraints
        transforms = torch.distributions.transforms
        constraint = constraints.independent(constraints.positive, 1)
        t = torch.distributions.constraint_registry.biject_to(constraint)
        self.assertIsInstance(t, transforms.IndependentTransform)
        self.assertIs(t.bijective, True)
        torch.manual_seed(102)
        x = torch.randn(2, 3, dtype=torch.float64)
        torch.testing.assert_close(t(x), torch.exp(x))

    def test_unregistered_constraint_raises(self):
        """Constraints without a biject_to entry must raise NotImplementedError."""
        constraints = torch.distributions.constraints
        constraint = constraints.MixtureSameFamilyConstraint(constraints.real)
        with self.assertRaisesRegex(NotImplementedError, "Cannot transform"):
            torch.distributions.constraint_registry.biject_to(constraint)

    def test_registered_transforms_are_bijective(self):
        """Every biject_to entry must advertise bijective=True (module contract)."""
        constraints = torch.distributions.constraints
        biject_to = torch.distributions.constraint_registry.biject_to
        for constraint in (
            constraints.real,
            constraints.positive,
            constraints.greater_than(1.0),
            constraints.greater_than_eq(1.0),
            constraints.less_than(1.0),
            constraints.unit_interval,
            constraints.half_open_interval(-2.0, 3.0),
            constraints.simplex,
        ):
            self.assertIs(biject_to(constraint).bijective, True)


if __name__ == "__main__":
    run_tests()
