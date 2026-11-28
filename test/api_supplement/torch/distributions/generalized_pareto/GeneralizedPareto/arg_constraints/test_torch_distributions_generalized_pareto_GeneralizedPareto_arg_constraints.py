# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.generalized_pareto.GeneralizedPareto.arg_constraints 接口功能正确性
API 名称：torch.distributions.generalized_pareto.GeneralizedPareto.arg_constraints
API 签名：arg_constraints = {"loc": constraints.real,
         "scale": constraints.positive,
         "concentration": constraints.real}（类属性 dict；源码 generalized_pareto.py:39-43）

覆盖维度表：
| 覆盖维度     | 说明                                              | 覆盖情况                                  |
|--------------|---------------------------------------------------|-------------------------------------------|
| 空/非空      | dict 非空且恰含三个键（无缺键/多键）              | 已覆盖（keys_match_source）               |
| 枚举选项     | 键名为 loc / scale / concentration 三个固定选项   | 已覆盖（keys_match_source）               |
| 参数类型     | 值分别为 real / positive / real 的 Constraint 对象| 已覆盖（values_are_constraint_singletons）|
| 传参与不传参 | 类属性形态，无入参                                | 已覆盖（keys_match_source）               |
| 等价类/边界  | 边界：scale=0（不含）、NaN、±inf                  | 已覆盖（check_boundaries）                |
| 正常传参     | 合法参数 check 为 True                            | 已覆盖（check_valid_invalid_params）      |
| 异常传参     | scale<=0 触发构造期 ValueError                    | 已覆盖（validate_args_rejects_bad_scale） |
| 混合设备类型 | 约束 dict 为纯 Python 结构，无设备迁移语义        | 不适用                                    |

未覆盖项及原因：
- 无。

注意：键与约束对象均与源码逐一对照（assertIs 单例同一性），check 边界
     （positive 开区间、real 拒绝 NaN）精确断言。
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
class TestGeneralizedParetoArgConstraints(TestCase):
    """Contract tests for GeneralizedPareto.arg_constraints."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_keys_match_source(self):
        """The dict must contain exactly the three parameter names from source."""
        expected_keys = {"loc", "scale", "concentration"}
        arg_constraints = (
            torch.distributions.GeneralizedPareto.arg_constraints
        )
        self.assertEqual(set(arg_constraints), expected_keys)
        dist = torch.distributions.GeneralizedPareto(
            0.0, 1.0, 0.5, validate_args=False
        )
        self.assertEqual(set(dist.arg_constraints), expected_keys)

    def test_values_are_constraint_singletons(self):
        """Entries must be the exact constraints.real / constraints.positive singletons."""
        constraints = torch.distributions.constraints
        arg_constraints = (
            torch.distributions.GeneralizedPareto.arg_constraints
        )
        self.assertIs(arg_constraints["loc"], constraints.real)
        self.assertIs(arg_constraints["scale"], constraints.positive)
        self.assertIs(arg_constraints["concentration"], constraints.real)

    def test_check_valid_invalid_params(self):
        """loc/concentration accept any finite value; scale must be > 0."""
        arg_constraints = (
            torch.distributions.GeneralizedPareto.arg_constraints
        )
        for name in ("loc", "concentration"):
            constraint = arg_constraints[name]
            finite = torch.tensor([-1e10, 0.0, 1e10])
            self.assertIs(constraint.check(finite).all().item(), True)
        scale = arg_constraints["scale"]
        self.assertIs(scale.check(torch.tensor([0.1, 2.0])).all().item(), True)
        self.assertIs(scale.check(torch.tensor(0.0)).item(), False)
        self.assertIs(scale.check(torch.tensor(-0.5)).item(), False)

    def test_check_boundaries(self):
        """real rejects NaN; positive is an open interval that rejects 0."""
        arg_constraints = (
            torch.distributions.GeneralizedPareto.arg_constraints
        )
        self.assertIs(
            arg_constraints["loc"].check(torch.tensor(float("nan"))).item(),
            False,
        )
        self.assertIs(
            arg_constraints["concentration"]
            .check(torch.tensor(float("inf")))
            .item(),
            True,
        )
        self.assertIs(
            arg_constraints["scale"].check(torch.tensor(0.0)).item(), False
        )

    def test_validate_args_rejects_bad_scale(self):
        """Construction with validate_args=True must reject scale <= 0."""
        with self.assertRaises(ValueError):
            torch.distributions.GeneralizedPareto(
                0.0, 0.0, 0.5, validate_args=True
            )
        with self.assertRaises(ValueError):
            torch.distributions.GeneralizedPareto(
                torch.tensor([0.0]),
                torch.tensor([-2.0]),
                torch.tensor([0.5]),
                validate_args=True,
            )


if __name__ == "__main__":
    run_tests()
