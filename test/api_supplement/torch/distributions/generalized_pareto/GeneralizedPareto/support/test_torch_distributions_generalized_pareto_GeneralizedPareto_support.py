# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.generalized_pareto.GeneralizedPareto.support 接口功能正确性
API 名称：torch.distributions.generalized_pareto.GeneralizedPareto.support
API 签名：@constraints.dependent_property(is_discrete=False, event_dim=0)
         def support(self) -> constraints.interval(loc, ...)（源码 generalized_pareto.py:145-152）

覆盖维度表：
| 覆盖维度     | 说明                                              | 覆盖情况                                  |
|--------------|---------------------------------------------------|-------------------------------------------|
| 空/非空      | support 为非 None 的 per-instance 约束对象        | 已覆盖（is_dependent_property_marker）    |
| 枚举选项     | conc>0（上界 inf）与 conc<0（有限上界）两选项     | 已覆盖（conc_positive/negative 两用例）   |
| 参数类型     | 无入参；返回 Constraint（interval）对象           | 已覆盖（conc_* 各用例）                   |
| 传参与不传参 | 属性读取形态；check 必传 value                    | 已覆盖（各用例）                          |
| 等价类/边界  | 边界：下界 loc 含在支持集内、上界闭区间           | 已覆盖（negative_finite_upper 等）        |
| 正常传参     | 固定种子采样全部落在 per-instance 支持集内        | 已覆盖（samples_inside_support）          |
| 异常传参     | 支持集外取值 check 返回 False（不抛异常）         | 已覆盖（conc_positive/negative 用例）     |
| 混合设备类型 | 约束对象为纯 Python 结构，无设备迁移语义          | 不适用                                    |

未覆盖项及原因：
- 无。

注意：support 依赖参数（dependent contract），须断言「不同参数实例返回
     不同区间」这一逐实例求值语义。
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
class TestGeneralizedParetoSupport(TestCase):
    """Contract tests for GeneralizedPareto.support."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_is_dependent_property_marker(self):
        """On the class, support is a dependent-property marker (undecidable)."""
        constraints = torch.distributions.constraints
        self.assertIs(
            constraints.is_dependent(
                torch.distributions.GeneralizedPareto.support
            ),
            True,
        )
        # On an instance it evaluates to a real interval constraint.
        dist = torch.distributions.GeneralizedPareto(0.0, 1.0, 0.5)
        self.assertIsInstance(dist.support, constraints._Interval)
        self.assertIs(dist.support.is_discrete, False)
        self.assertEqual(dist.support.event_dim, 0)

    def test_conc_positive_open_upper(self):
        """conc>0: lower bound is loc and the upper bound admits large values."""
        constraints = torch.distributions.constraints
        dist = torch.distributions.GeneralizedPareto(
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(2.0, dtype=torch.float64),
            torch.tensor(0.5, dtype=torch.float64),
        )
        support = dist.support
        torch.testing.assert_close(
            torch.as_tensor(support.lower_bound),
            torch.tensor(1.0, dtype=torch.float64),
        )
        values = torch.tensor([0.5, 1.0, 1.5, 1e6], dtype=torch.float64)
        result = support.check(values)
        self.assertEqual(
            result.tolist(), [False, True, True, True]
        )

    def test_conc_negative_finite_upper(self):
        """conc<0: support is [loc, loc - scale/conc] with closed bounds."""
        dist = torch.distributions.GeneralizedPareto(
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(-0.5, dtype=torch.float64),
        )
        support = dist.support
        torch.testing.assert_close(
            torch.as_tensor(support.upper_bound),
            torch.tensor(2.0, dtype=torch.float64),
        )
        values = torch.tensor([-0.1, 0.0, 2.0, 2.5], dtype=torch.float64)
        result = support.check(values)
        self.assertEqual(
            result.tolist(), [False, True, True, False]
        )

    def test_support_adapts_per_instance(self):
        """Support depends on the instance's parameters (dependent contract)."""
        dist_a = torch.distributions.GeneralizedPareto(1.0, 1.0, 0.5)
        dist_b = torch.distributions.GeneralizedPareto(3.0, 1.0, 0.5)
        self.assertIs(dist_a.support.check(torch.tensor(2.0)).item(), True)
        self.assertIs(dist_b.support.check(torch.tensor(2.0)).item(), False)

    def test_samples_inside_support(self):
        """Seeded samples must all satisfy the per-instance support check."""
        torch.manual_seed(41)
        dist = torch.distributions.GeneralizedPareto(
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(1.5, dtype=torch.float64),
            torch.tensor(-0.5, dtype=torch.float64),
        )
        samples = dist.sample((2000,))
        self.assertEqual(samples.shape, (2000,))
        self.assertEqual(dist.support.check(samples).all().item(), True)


if __name__ == "__main__":
    run_tests()
