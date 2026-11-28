# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.geometric.Geometric.arg_constraints 接口功能正确性
API 名称：torch.distributions.geometric.Geometric.arg_constraints
API 签名：arg_constraints = {"probs": constraints.unit_interval,
         "logits": constraints.real}（类属性 dict；源码 geometric.py:46-47）

覆盖维度表：
| 覆盖维度     | 说明                                              | 覆盖情况                                  |
|--------------|---------------------------------------------------|-------------------------------------------|
| 空/非空      | dict 非空且恰含两个键（无缺键/多键）              | 已覆盖（keys_match_source）               |
| 枚举选项     | 键名为 probs / logits 两个固定选项                | 已覆盖（keys_match_source）               |
| 参数类型     | 值为 unit_interval / real 的 Constraint 单例      | 已覆盖（values_are_constraint_singletons）|
| 传参与不传参 | 类属性形态，无入参                                | 已覆盖（keys_match_source）               |
| 等价类/边界  | 边界：probs=0/1（闭区间含端点）、越界 -0.1/1.1    | 已覆盖（check_valid_invalid_params）      |
| 正常传参     | 合法参数 check 为 True                            | 已覆盖（check_valid_invalid_params）      |
| 异常传参     | probs=0 触发构造期 ValueError（超出 unit_interval | 已覆盖（validate_args_extra_positive）    |
|              | 的额外正性校验，源码 geometric.py:76-87）         |                                           |
| 混合设备类型 | 约束 dict 为纯 Python 结构，无设备迁移语义        | 不适用                                    |

未覆盖项及原因：
- 无。

注意：Geometric 对 probs 的约束语义比 unit_interval 更严（须 >0）：本文件
     同时断言 unit_interval 闭区间端点 check=True 与构造期 probs=0 报错，
     两者并不矛盾而是源码的分层校验契约。
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
class TestGeometricArgConstraints(TestCase):
    """Contract tests for torch.distributions.geometric.Geometric.arg_constraints."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_keys_match_source(self):
        """The dict must contain exactly the two parameter names from source."""
        expected_keys = {"probs", "logits"}
        self.assertEqual(
            set(torch.distributions.Geometric.arg_constraints), expected_keys
        )
        dist = torch.distributions.Geometric(
            probs=torch.tensor([0.3]), validate_args=False
        )
        self.assertEqual(set(dist.arg_constraints), expected_keys)

    def test_values_are_constraint_singletons(self):
        """Entries must be the exact unit_interval / real singleton objects."""
        constraints = torch.distributions.constraints
        arg_constraints = torch.distributions.Geometric.arg_constraints
        self.assertIs(arg_constraints["probs"], constraints.unit_interval)
        self.assertIs(arg_constraints["logits"], constraints.real)

    def test_check_valid_invalid_params(self):
        """unit_interval is closed [0, 1]: endpoints True, outside False."""
        arg_constraints = torch.distributions.Geometric.arg_constraints
        probs = arg_constraints["probs"]
        valid = torch.tensor([0.0, 0.3, 1.0])
        self.assertIs(probs.check(valid).all().item(), True)
        outside = torch.tensor([-0.1, 1.1])
        self.assertIs(probs.check(outside).any().item(), False)
        self.assertIs(probs.check(torch.tensor(1.5)).item(), False)
        # logits accepts any finite value.
        logits = arg_constraints["logits"]
        self.assertIs(
            logits.check(torch.tensor([-10.0, 0.0, 10.0])).all().item(), True
        )

    def test_validate_args_extra_positive_check(self):
        """Beyond unit_interval, Geometric requires probs > 0 at construction."""
        # probs = 0 satisfies unit_interval but fails the extra positivity check.
        with self.assertRaisesRegex(ValueError, "to be positive"):
            torch.distributions.Geometric(probs=0.0, validate_args=True)
        with self.assertRaisesRegex(ValueError, "to be positive"):
            torch.distributions.Geometric(
                probs=torch.tensor([0.0, 0.5]), validate_args=True
            )
        # probs = 1 and interior values pass both layers.
        torch.distributions.Geometric(probs=1.0, validate_args=True)
        torch.distributions.Geometric(
            probs=torch.tensor([0.2, 0.9]), validate_args=True
        )
        # Opting out is honored: probs = 0 constructs without validation.
        torch.distributions.Geometric(probs=0.0, validate_args=False)

    def test_probs_and_logits_forms_share_constraints(self):
        """Both parameterization forms expose the same arg_constraints dict."""
        by_probs = torch.distributions.Geometric(
            probs=torch.tensor([0.3]), validate_args=False
        )
        by_logits = torch.distributions.Geometric(
            logits=torch.tensor([0.1]), validate_args=False
        )
        self.assertIs(
            by_probs.arg_constraints, torch.distributions.Geometric.arg_constraints
        )
        self.assertIs(
            by_logits.arg_constraints,
            torch.distributions.Geometric.arg_constraints,
        )


if __name__ == "__main__":
    run_tests()
