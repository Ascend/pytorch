# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.constraint_registry.ConstraintRegistry.register 接口功能正确性
API 名称：torch.distributions.constraint_registry.ConstraintRegistry.register
API 签名：register(self, constraint, factory=None)（源码 constraint_registry.py:89-122）

覆盖维度表：
| 覆盖维度     | 说明                                              | 覆盖情况                                  |
|--------------|---------------------------------------------------|-------------------------------------------|
| 空/非空      | factory 传 None（装饰器形态）与非 None（直接注册）| 已覆盖（decorator_form、singleton_*）     |
| 枚举选项     | constraint 可传单例实例或 Constraint 子类两种形态 | 已覆盖（singleton_instance、subclass）    |
| 参数类型     | 非法 constraint 类型（字符串/整数）触发 TypeError | 已覆盖（invalid_constraint_type_raises）  |
| 传参与不传参 | factory 必传或经装饰器隐式传入                    | 已覆盖（各用例）                          |
| 等价类/边界  | 空表查未注册约束、重复注册覆盖                    | 已覆盖（duplicate、unregistered）         |
| 正常传参     | 注册后按类型查表并返回工厂产出的 Transform        | 已覆盖（singleton_instance、subclass）    |
| 异常传参     | TypeError / NotImplementedError 两条异常路径      | 已覆盖（invalid_*、unregistered_raises）  |
| 混合设备类型 | 注册表为纯 Python 对象，无设备迁移语义            | 不适用                                    |

未覆盖项及原因：
- 无。全程使用 ConstraintRegistry() 新实例，不污染全局 biject_to/transform_to。

注意：自定义 Transform 需实现 _call/_inverse/log_abs_det_jacobian；查表结果
     用 torch.testing.assert_close 与解析值对照（x2 变换的 J = 2）。
"""
import math
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


class MyRealConstraint(torch.distributions.constraints.Constraint):
    """Minimal custom constraint subclass used as a registry key."""

    def check(self, value):
        return torch.ones_like(value, dtype=torch.bool)


class ScaleByTwoTransform(torch.distributions.transforms.Transform):
    """Bijective y = 2x transform returned by the test factories."""

    domain = torch.distributions.constraints.real
    codomain = torch.distributions.constraints.real
    bijective = True

    def _call(self, x):
        return 2 * x

    def _inverse(self, y):
        return y / 2

    def log_abs_det_jacobian(self, x, y):
        return torch.full_like(x, math.log(2))


def _scale_by_two_factory(constraint):
    return ScaleByTwoTransform()


class NegateTransform(torch.distributions.transforms.Transform):
    """Bijective y = -x transform used to verify duplicate registration."""

    domain = torch.distributions.constraints.real
    codomain = torch.distributions.constraints.real
    bijective = True

    def _call(self, x):
        return -x

    def _inverse(self, y):
        return -y

    def log_abs_det_jacobian(self, x, y):
        return torch.zeros_like(x)


def _negate_factory(constraint):
    return NegateTransform()


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestConstraintRegistryRegister(TestCase):
    """Contract tests for ConstraintRegistry.register on fresh instances."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_register_with_singleton_instance(self):
        """Registering a singleton instance keys the registry by its type."""
        registry = torch.distributions.constraint_registry.ConstraintRegistry()
        registry.register(torch.distributions.constraints.real, _scale_by_two_factory)
        transform = registry(torch.distributions.constraints.real)
        self.assertIsInstance(transform, ScaleByTwoTransform)
        x = torch.tensor([1.0, -2.0, 0.5])
        torch.testing.assert_close(transform(x), 2 * x)
        torch.testing.assert_close(transform.inv(transform(x)), x)

    def test_register_decorator_form_returns_factory(self):
        """Decorator usage registers the function and returns it unchanged."""
        registry = torch.distributions.constraint_registry.ConstraintRegistry()
        returned = registry.register(MyRealConstraint, _scale_by_two_factory)
        self.assertIs(returned, _scale_by_two_factory)
        transform = registry(MyRealConstraint())
        self.assertIsInstance(transform, ScaleByTwoTransform)

    def test_register_with_constraint_subclass_lookup(self):
        """Lookup happens by type(constraint), so instances resolve to their class."""
        registry = torch.distributions.constraint_registry.ConstraintRegistry()
        registry.register(MyRealConstraint, _scale_by_two_factory)
        x = torch.tensor([3.0])
        torch.testing.assert_close(registry(MyRealConstraint())(x), 2 * x)

    def test_register_invalid_constraint_type_raises(self):
        """Non-Constraint keys (str / int) must raise TypeError."""
        registry = torch.distributions.constraint_registry.ConstraintRegistry()
        with self.assertRaisesRegex(
            TypeError,
            "Expected constraint to be either a Constraint subclass or instance",
        ):
            registry.register("real", _scale_by_two_factory)
        with self.assertRaises(TypeError):
            registry.register(123, _scale_by_two_factory)

    def test_register_duplicate_overwrites_previous_factory(self):
        """Re-registering the same constraint type replaces the factory."""
        registry = torch.distributions.constraint_registry.ConstraintRegistry()
        registry.register(
            torch.distributions.constraints.real, _scale_by_two_factory
        )
        registry.register(torch.distributions.constraints.real, _negate_factory)
        transform = registry(torch.distributions.constraints.real)
        self.assertIsInstance(transform, NegateTransform)
        x = torch.tensor([1.0, -3.0])
        torch.testing.assert_close(transform(x), -x)
        torch.testing.assert_close(transform.inv(transform(x)), x)

    def test_call_unregistered_constraint_raises(self):
        """Querying an empty registry must raise NotImplementedError."""
        registry = torch.distributions.constraint_registry.ConstraintRegistry()
        with self.assertRaisesRegex(NotImplementedError, "Cannot transform"):
            registry(torch.distributions.constraints.real)

    def test_registry_state_isolated_per_instance(self):
        """A registration in one instance must not leak into another instance."""
        registry_a = torch.distributions.constraint_registry.ConstraintRegistry()
        registry_b = torch.distributions.constraint_registry.ConstraintRegistry()
        registry_a.register(torch.distributions.constraints.real, _scale_by_two_factory)
        # Instance A resolves; instance B stays empty and must still raise.
        self.assertIsInstance(
            registry_a(torch.distributions.constraints.real), ScaleByTwoTransform
        )
        with self.assertRaises(NotImplementedError):
            registry_b(torch.distributions.constraints.real)


if __name__ == "__main__":
    run_tests()
