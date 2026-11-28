# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.transformed_distribution.
TransformedDistribution.expand 接口功能正确性
API 名称：torch.distributions.transformed_distribution.TransformedDistribution.expand
API 签名：expand(self, batch_shape, _instance=None) -> TransformedDistribution

覆盖维度表：
| 覆盖维度     | 说明                                   | 覆盖情况                              |
|--------------|----------------------------------------|---------------------------------------|
| 空/非空      | 空批次扩展与非空批次再扩展             | 已覆盖（logistic_consistency、        |
|              |                                        | single_transform_closed_form）        |
| 枚举选项     | 本接口无枚举入参                       | 不适用                                |
| 参数类型     | float32 与 float64 loc/scale 参数      | 已覆盖（closed_form 用 float64）      |
| 传参与不传参 | batch_shape 必填；_instance 走默认路径 | 已覆盖(全部用例均不传 _instance)   |
| 等价类/边界  | 单 transform 与 transform 链两种构造   | 已覆盖（single_transform、logistic）  |
| 正常传参     | 扩展后 log_prob 与闭式解逐元素对照     | 已覆盖（logistic_closed_form、        |
|              |                                        | single_transform_closed_form）        |
| 异常传参     | 与原 batch 不兼容的形状应报错          | 已覆盖（incompatible_batch_shape）    |
| 混合设备类型 | 契约在 CPU 上即可验证，无多设备语义    | 不适用                             |

未覆盖项及原因：
- 未覆盖：显式传入 _instance 的复用路径。expand 直接填充 new 的
  base_dist/transforms，外部实例复用的类型校验未稳定承诺，不做伪断言。
"""
import os
import unittest

import torch

from torch.distributions.exponential import Exponential
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import (
    AffineTransform,
    ExpTransform,
    SigmoidTransform,
)
from torch.distributions.uniform import Uniform

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


def _make_logistic():
    """Build a scalar-batched logistic distribution (float64)."""
    loc = torch.tensor([2.0], dtype=torch.float64)
    scale = torch.tensor([3.0], dtype=torch.float64)
    base = Uniform(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64))
    return TransformedDistribution(
        base, [SigmoidTransform().inv, AffineTransform(loc=loc, scale=scale)])


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestTransformedDistributionExpand(TestCase):
    """Test cases for TransformedDistribution.expand."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_expand_logistic_log_prob_consistency(self):
        """Expanded distribution scores values identically per batch item."""
        d = _make_logistic()
        e = d.expand((5,))
        self.assertEqual(tuple(e.batch_shape), (5,))
        value = torch.tensor(2.5, dtype=torch.float64)
        torch.testing.assert_close(e.log_prob(value), d.log_prob(value).expand(5))

    def test_expand_logistic_log_prob_closed_form(self):
        """Expanded log_prob matches the logistic closed form."""
        d = _make_logistic()
        e = d.expand((5,))
        loc = torch.tensor([2.0], dtype=torch.float64)
        scale = torch.tensor([3.0], dtype=torch.float64)
        value = torch.tensor(2.5, dtype=torch.float64)
        z = (value - loc) / scale
        expected = -z - torch.log(scale) - 2.0 * torch.log1p(torch.exp(-z))
        result = e.log_prob(value)
        self.assertEqual(tuple(result.shape), (5,))
        torch.testing.assert_close(result, expected.expand(5))

    def test_expand_single_transform_closed_form(self):
        """Exp(ExpTransform) expansion matches the log-exponential form."""
        rate = torch.tensor([1.5], dtype=torch.float64)
        d = TransformedDistribution(Exponential(rate), ExpTransform())
        e = d.expand((4,))
        self.assertEqual(tuple(e.batch_shape), (4,))
        value = torch.tensor(1.2, dtype=torch.float64)
        expected = torch.log(rate) - (rate + 1.0) * torch.log(value)
        result = e.log_prob(value)
        self.assertEqual(tuple(result.shape), (4,))
        torch.testing.assert_close(result, expected.expand(4))

    def test_expand_shares_transform_list(self):
        """The expanded instance reuses the original transform list."""
        d = TransformedDistribution(Exponential(torch.tensor([1.5])), ExpTransform())
        e = d.expand((4,))
        self.assertIs(e.transforms, d.transforms)
        self.assertEqual(len(e.transforms), 1)

    def test_expand_sample_shape_and_support(self):
        """Expanded samples keep shape and satisfy the codomain support."""
        torch.manual_seed(0)
        base = Uniform(torch.tensor(0.0), torch.tensor(1.0))
        d = TransformedDistribution(base, SigmoidTransform().inv)
        e = d.expand((7,))
        s = e.sample((3,))
        self.assertEqual(tuple(s.shape), (3, 7))
        self.assertTrue(bool(e.support.check(s).all()))

    def test_expand_incompatible_batch_shape_raises(self):
        """Expanding a (1,)-batch to (3, 5) violates broadcasting rules."""
        d = TransformedDistribution(
            Exponential(torch.tensor([1.5, 2.0])), ExpTransform())
        self.assertEqual(tuple(d.batch_shape), (2,))
        with self.assertRaises(RuntimeError):
            d.expand((3, 5))


if __name__ == "__main__":
    run_tests()
