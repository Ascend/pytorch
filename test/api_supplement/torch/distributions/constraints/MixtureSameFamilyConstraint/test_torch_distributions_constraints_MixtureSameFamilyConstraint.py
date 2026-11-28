# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.constraints.MixtureSameFamilyConstraint 接口功能正确性
API 名称：torch.distributions.constraints.MixtureSameFamilyConstraint
API 签名：MixtureSameFamilyConstraint(base_constraint: Constraint)
         （源码 constraints.py:280-326）

覆盖维度表：
| 覆盖维度     | 说明                                              | 覆盖情况                                  |
|--------------|---------------------------------------------------|-------------------------------------------|
| 空/非空      | base_constraint 传 None/非 Constraint 为空参场景  | 已覆盖（init_requires_constraint）        |
| 枚举选项     | 可包裹 real/boolean/independent 等各约束类型      | 已覆盖（delegation 各用例）               |
| 参数类型     | 非 Constraint 类型入参触发 AssertionError         | 已覆盖（init_requires_constraint）        |
| 传参与不传参 | 单参形态，base_constraint 必传                    | 已覆盖（init_accepts_constraint）         |
| 等价类/边界  | 离散/连续、event_dim 0/N 的约束等价类             | 已覆盖（delegation_is_discrete、_dim）    |
| 正常传参     | 构造成功且 base_constraint 原样保留               | 已覆盖（init_accepts_constraint）         |
| 异常传参     | 非法 base_constraint 抛 AssertionError            | 已覆盖（init_requires_constraint）        |
| 混合设备类型 | 约束对象为纯 Python 结构，无设备迁移语义          | 不适用                                    |

未覆盖项及原因：
- 无。check() 的行为契约由独立文件
  test_torch_distributions_constraints_MixtureSameFamilyConstraint_check.py 覆盖。

注意：断言落在「构造校验 / 属性委托（is_discrete、event_dim）/ repr 结构」
     三类可证伪契约上。
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


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestMixtureSameFamilyConstraint(TestCase):
    """Contract tests for constraints.MixtureSameFamilyConstraint."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_init_accepts_constraint_and_keeps_base(self):
        """A valid Constraint instance is stored verbatim as base_constraint."""
        constraints = torch.distributions.constraints
        constraint = constraints.MixtureSameFamilyConstraint(constraints.real)
        self.assertIsInstance(constraint, constraints.Constraint)
        self.assertIs(constraint.base_constraint, constraints.real)

    def test_init_requires_constraint_instance(self):
        """Non-Constraint base_constraint must fail with AssertionError."""
        constraints = torch.distributions.constraints
        for bad in ("real", 123, None, [constraints.real]):
            with self.assertRaisesRegex(
                AssertionError, "base_constraint must be a Constraint"
            ):
                constraints.MixtureSameFamilyConstraint(bad)

    def test_is_discrete_delegates_to_base(self):
        """is_discrete must mirror the base constraint's discreteness."""
        constraints = torch.distributions.constraints
        discrete = constraints.MixtureSameFamilyConstraint(constraints.boolean)
        continuous = constraints.MixtureSameFamilyConstraint(constraints.real)
        self.assertIs(discrete.is_discrete, True)
        self.assertIs(continuous.is_discrete, False)

    def test_event_dim_delegates_to_base(self):
        """event_dim must mirror the base constraint (incl. independent wrapper)."""
        constraints = torch.distributions.constraints
        zero_dim = constraints.MixtureSameFamilyConstraint(constraints.real)
        self.assertEqual(zero_dim.event_dim, 0)
        base = constraints.independent(constraints.real, 2)
        nested = constraints.MixtureSameFamilyConstraint(base)
        self.assertEqual(nested.event_dim, 2)

    def test_repr_contains_class_and_base(self):
        """repr() must expose the wrapper class and the base constraint."""
        constraints = torch.distributions.constraints
        constraint = constraints.MixtureSameFamilyConstraint(constraints.positive)
        text = repr(constraint)
        self.assertIn("MixtureSameFamilyConstraint", text)
        self.assertIn(repr(constraints.positive), text)


if __name__ == "__main__":
    run_tests()
