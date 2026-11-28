# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.beta.Beta.expand 接口功能正确性
API 名称：torch.distributions.beta.Beta.expand
API 签名：expand(self, batch_shape, _instance=None) -> Beta（源码 beta.py:63-69）

覆盖维度表：
| 覆盖维度     | 说明                                            | 覆盖情况                              |
|--------------|-------------------------------------------------|---------------------------------------|
| 空/非空      | batch_shape 为 () 与非空元组两种形态            | 已覆盖（scalar_to_batch、same_shape） |
| 枚举选项     | 本接口无枚举入参                                | 不适用                                |
| 参数类型     | batch_shape 接受 tuple / torch.Size             | 已覆盖（log_prob_matches_original）   |
| 传参与不传参 | batch_shape 必传；_instance 省略用默认 None     | 已覆盖（各用例均省略 _instance）      |
| 等价类/边界  | 同形展开、缩小 batch 的非法展开                 | 已覆盖（same_shape、smaller_raises）  |
| 正常传参     | 扩展后 batch_shape/参数/密度逐元素与原分布一致  | 已覆盖（log_prob_matches_original）   |
| 异常传参     | 缩小 batch_shape 触发 RuntimeError              | 已覆盖（smaller_batch_raises）        |
| 混合设备类型 | 参数为 CPU 张量，纯分布展开无设备迁移语义       | 不适用                                |

未覆盖项及原因：
- 未覆盖：显式传入 _instance 复用已有实例的路径。该参数是子类 expand 内部
  递归使用的私有协议（_get_checked_instance），公开契约以默认 None 为准。

注意：浮点对照一律使用 torch.testing.assert_close：扩展后 log_prob 与手工
     闭式解（Beta 密度对数）对照，其余与原分布广播结果对照。
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


def _beta_log_prob(x, alpha, beta):
    """Closed-form log density of the Beta distribution (torch double ops)."""
    return (
        (alpha - 1.0) * torch.log(x)
        + (beta - 1.0) * torch.log(1.0 - x)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
        + torch.lgamma(alpha + beta)
    )


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestBetaExpand(TestCase):
    """Contract tests for torch.distributions.beta.Beta.expand."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_expand_batch_shape_and_params(self):
        """Expanded instance reports the target batch shape and expanded params."""
        dist = torch.distributions.Beta(
            torch.tensor([2.0, 3.0], dtype=torch.float64),
            torch.tensor([0.5, 1.5], dtype=torch.float64),
        )
        expanded = dist.expand((4, 2))
        self.assertIsInstance(expanded, torch.distributions.Beta)
        self.assertEqual(expanded.batch_shape, torch.Size((4, 2)))
        self.assertEqual(expanded.concentration1.shape, (4, 2))
        self.assertEqual(expanded.concentration0.shape, (4, 2))
        torch.testing.assert_close(
            expanded.concentration1,
            dist.concentration1.expand((4, 2)),
        )
        torch.testing.assert_close(
            expanded.concentration0,
            dist.concentration0.expand((4, 2)),
        )

    def test_expand_log_prob_matches_original_and_closed_form(self):
        """log_prob after expand must equal the original (broadcast) and the
        closed-form Beta density, elementwise, on the target batch shape."""
        dist = torch.distributions.Beta(
            torch.tensor([2.0, 3.0], dtype=torch.float64),
            torch.tensor([0.5, 1.5], dtype=torch.float64),
        )
        expanded = dist.expand((4, 2))
        x = torch.linspace(0.05, 0.95, 8, dtype=torch.float64).reshape(4, 2)
        actual = expanded.log_prob(x)
        self.assertEqual(actual.shape, (4, 2))
        torch.testing.assert_close(actual, dist.log_prob(x))
        alpha = expanded.concentration1
        beta = expanded.concentration0
        torch.testing.assert_close(actual, _beta_log_prob(x, alpha, beta))

    def test_expand_scalar_to_batch(self):
        """A scalar-parameter Beta can be expanded to a non-empty batch."""
        dist = torch.distributions.Beta(2.0, 3.0)
        self.assertEqual(dist.batch_shape, torch.Size())
        expanded = dist.expand(torch.Size((5,)))
        self.assertEqual(expanded.batch_shape, torch.Size((5,)))
        torch.manual_seed(11)
        samples = expanded.sample((3,))
        self.assertEqual(samples.shape, (3, 5))
        # Seeded samples must lie inside the Beta support (0, 1).
        inside = (samples > 0.0) & (samples < 1.0)
        self.assertEqual(inside.all().item(), True)

    def test_expand_mean_variance_match_original(self):
        """Derived moments after expand must match the original broadcast."""
        dist = torch.distributions.Beta(
            torch.tensor([2.0, 4.0], dtype=torch.float64),
            torch.tensor([3.0, 5.0], dtype=torch.float64),
        )
        expanded = dist.expand((3, 2))
        torch.testing.assert_close(
            expanded.mean, dist.mean.expand((3, 2))
        )
        torch.testing.assert_close(
            expanded.variance, dist.variance.expand((3, 2))
        )

    def test_expand_same_shape_is_equivalent(self):
        """Expanding to the current batch shape keeps log_prob identical."""
        dist = torch.distributions.Beta(
            torch.tensor([2.0, 3.0], dtype=torch.float64),
            torch.tensor([0.5, 1.5], dtype=torch.float64),
        )
        expanded = dist.expand((2,))
        self.assertEqual(expanded.batch_shape, torch.Size((2,)))
        x = torch.tensor([0.2, 0.8], dtype=torch.float64)
        torch.testing.assert_close(expanded.log_prob(x), dist.log_prob(x))

    def test_expand_smaller_batch_raises(self):
        """Expanding a batch dim to a smaller size must fail (tensor expand rule)."""
        dist = torch.distributions.Beta(
            torch.tensor([2.0, 3.0], dtype=torch.float64),
            torch.tensor([0.5, 1.5], dtype=torch.float64),
        )
        with self.assertRaises(RuntimeError):
            dist.expand((1,))


if __name__ == "__main__":
    run_tests()
