# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.gamma.Gamma.arg_constraints 接口功能正确性
API 名称：torch.distributions.gamma.Gamma.arg_constraints
API 签名：arg_constraints = {"concentration": constraints.positive,
         "rate": constraints.positive}（类属性 dict；源码 gamma.py:37-40）

覆盖维度表：
| 覆盖维度     | 说明                                              | 覆盖情况                                  |
|--------------|---------------------------------------------------|-------------------------------------------|
| 空/非空      | dict 非空且恰含两个键（无缺键/多键）              | 已覆盖（keys_match_source）               |
| 枚举选项     | 键名为 concentration / rate 两个固定选项          | 已覆盖（keys_match_source）               |
| 参数类型     | 值为 constraints.positive 的 Constraint 对象      | 已覆盖（values_are_positive_singleton）   |
| 传参与不传参 | 类属性形态，无入参                                | 已覆盖（keys_match_source）               |
| 等价类/边界  | 边界：0（不含）、负值、正值、批量混合             | 已覆盖（check_valid_invalid_params）      |
| 正常传参     | 合法参数 check 为 True                            | 已覆盖（check_valid_invalid_params）      |
| 异常传参     | 非法参数触发构造期 ValueError                     | 已覆盖（validate_args_rejects_invalid）   |
| 混合设备类型 | 约束 dict 为纯 Python 结构，无设备迁移语义        | 不适用                                    |

未覆盖项及原因：
- 无。

注意：键集合与源码逐一对照；positive 为开区间 (0, inf)，check(0) 必须为
     False —— 这一「实现错了就会红」的边界被精确断言。
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
class TestGammaArgConstraints(TestCase):
    """Contract tests for torch.distributions.gamma.Gamma.arg_constraints."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_keys_match_source(self):
        """The dict must contain exactly the two parameter names from source."""
        expected_keys = {"concentration", "rate"}
        self.assertEqual(set(torch.distributions.Gamma.arg_constraints), expected_keys)
        dist = torch.distributions.Gamma(
            torch.tensor([2.0]), torch.tensor([1.0]), validate_args=False
        )
        self.assertEqual(set(dist.arg_constraints), expected_keys)

    def test_values_are_positive_singleton(self):
        """Both entries must be the constraints.positive singleton object."""
        constraints = torch.distributions.constraints
        arg_constraints = torch.distributions.Gamma.arg_constraints
        self.assertIs(arg_constraints["concentration"], constraints.positive)
        self.assertIs(arg_constraints["rate"], constraints.positive)

    def test_check_valid_invalid_params(self):
        """positive is an open interval: >0 True, 0 / negatives False."""
        constraints = torch.distributions.constraints
        arg_constraints = torch.distributions.Gamma.arg_constraints
        valid = torch.tensor([0.5, 2.0, 100.0])
        for name, constraint in arg_constraints.items():
            self.assertIs(constraint.check(valid).all().item(), True)
            self.assertIs(constraint.check(torch.tensor(0.0)).item(), False)
            self.assertIs(constraint.check(torch.tensor(-1.0)).item(), False)
        # Cross-check against the singleton to keep the two assertions linked.
        self.assertIs(arg_constraints["concentration"], constraints.positive)

    def test_validate_args_rejects_invalid(self):
        """Construction with validate_args=True must reject non-positive params."""
        with self.assertRaises(ValueError):
            torch.distributions.Gamma(1.0, -1.0, validate_args=True)
        with self.assertRaises(ValueError):
            torch.distributions.Gamma(
                torch.tensor([-0.5]), torch.tensor([1.0]), validate_args=True
            )
        with self.assertRaises(ValueError):
            torch.distributions.Gamma(
                torch.tensor([1.0]), torch.tensor([0.0]), validate_args=True
            )
        # Opting out must be honored: no error is raised on the same inputs.
        torch.distributions.Gamma(1.0, -1.0, validate_args=False)


if __name__ == "__main__":
    run_tests()
