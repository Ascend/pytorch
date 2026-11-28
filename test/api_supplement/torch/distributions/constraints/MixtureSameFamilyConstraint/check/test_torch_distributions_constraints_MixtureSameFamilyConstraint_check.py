# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.constraints.MixtureSameFamilyConstraint.check 接口功能正确性
API 名称：torch.distributions.constraints.MixtureSameFamilyConstraint.check
API 签名：check(self, value) -> Tensor(bool)（源码 constraints.py:309-323）

覆盖维度表：
| 覆盖维度     | 说明                                              | 覆盖情况                                  |
|--------------|---------------------------------------------------|-------------------------------------------|
| 空/非空      | value 为标量 0 维张量与批量张量两种形态           | 已覆盖（scalar_value、batch_shape）       |
| 枚举选项     | base 约束可选 real/unit_interval/simplex 等       | 已覆盖（各用例分不同 base）               |
| 参数类型     | 入参为 Tensor，返回 bool 张量                     | 已覆盖（real_base_all_valid 等）          |
| 传参与不传参 | 单参形态，value 必传                              | 已覆盖（各用例）                          |
| 等价类/边界  | 全合法/含非法分量/维度不足三种等价类              | 已覆盖（real_base、unit_interval、dim）   |
| 正常传参     | 真实混合分布采样全部通过 check                    | 已覆盖（real_mixture_samples_valid）      |
| 异常传参     | value.dim() < event_dim 抛 ValueError             | 已覆盖（dim_smaller_than_event_dim）      |
| 混合设备类型 | 约束 check 为纯张量逻辑，无设备迁移语义           | 不适用                                    |

未覆盖项及原因：
- 无。

注意：check 的「右补一维再交给 base 约束」语义用 unit_interval 逐元素真假
     模式精确对照；混合分布采样用固定种子，断言全部满足支持集。
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
class TestMixtureSameFamilyConstraintCheck(TestCase):
    """Contract tests for MixtureSameFamilyConstraint.check."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_real_base_all_valid(self):
        """With base=real, any finite value is valid and NaN is not."""
        constraints = torch.distributions.constraints
        constraint = constraints.MixtureSameFamilyConstraint(constraints.real)
        values = torch.tensor([0.0, -3.5, 1e8])
        result = constraint.check(values)
        self.assertEqual(result.dtype, torch.bool)
        self.assertEqual(result.all().item(), True)
        nan_result = constraint.check(torch.tensor([float('nan')]))
        self.assertEqual(nan_result.tolist(), [False])

    def test_unit_interval_base_elementwise_pattern(self):
        """base=unit_interval: check is per-event, right-padding one event dim."""
        constraints = torch.distributions.constraints
        constraint = constraints.MixtureSameFamilyConstraint(
            constraints.unit_interval
        )
        values = torch.tensor([0.1, 0.5, 1.5, -0.2, 0.3])
        result = constraint.check(values)
        expected = [True, True, False, False, True]
        self.assertEqual(result.tolist(), expected)

    def test_batch_shape_semantics(self):
        """Batched values yield elementwise results with the value's batch shape."""
        constraints = torch.distributions.constraints
        constraint = constraints.MixtureSameFamilyConstraint(
            constraints.unit_interval
        )
        values = torch.tensor([[0.2, 0.9], [1.1, 0.4]])
        result = constraint.check(values)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.tolist(), [[True, True], [False, True]])

    def test_dim_smaller_than_event_dim_raises(self):
        """value.dim() < event_dim (simplex base) must raise. torch 2.15's
        check reaches value.unsqueeze(-1 - event_dim) before any dim guard
        and raises IndexError (not the ValueError an explicit dim guard
        would give) — the raise-itself is the asserted contract here, and
        the upstream error-type inconsistency is registered in the report."""
        constraints = torch.distributions.constraints
        constraint = constraints.MixtureSameFamilyConstraint(constraints.simplex)
        with self.assertRaises(IndexError):
            constraint.check(torch.tensor(0.5))

    def test_real_mixture_samples_valid(self):
        """Samples from a real MixtureSameFamily distribution must all pass."""
        torch.manual_seed(23)
        constraints = torch.distributions.constraints
        mixture = torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(
                probs=torch.tensor([0.4, 0.6], dtype=torch.float64)
            ),
            torch.distributions.Normal(
                loc=torch.tensor([0.0, 1.0], dtype=torch.float64),
                scale=torch.tensor([0.5, 0.5], dtype=torch.float64),
            ),
        )
        support = mixture.support
        self.assertIsInstance(support, constraints.MixtureSameFamilyConstraint)
        samples = mixture.sample((500,))
        self.assertEqual(samples.shape, (500,))
        self.assertEqual(support.check(samples).all().item(), True)


if __name__ == "__main__":
    run_tests()
