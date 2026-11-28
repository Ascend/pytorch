# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.multinomial.Multinomial.sample 接口功能正确性
API 名称：torch.distributions.multinomial.Multinomial.sample
API 签名：sample(self, sample_shape=torch.Size()) -> Tensor

覆盖维度表：
| 覆盖维度     | 说明                                   | 覆盖情况                              |
|--------------|----------------------------------------|---------------------------------------|
| 空/非空      | 空样本形状与非空样本形状               | 已覆盖（default_shape、with_shape）   |
| 枚举选项     | 本接口无枚举入参                       | 不适用                                |
| 参数类型     | float32 与 float64 概率张量            | 已覆盖（dtype_matches_probs）         |
| 传参与不传参 | 不传参默认 torch.Size()；传 tuple 形状 | 已覆盖（default_shape、with_shape）   |
| 等价类/边界  | 批量参数与 logits 构造形态             | 已覆盖（batched、logits_path）        |
| 正常传参     | 固定种子后断言形状、计数总和与 support | 已覆盖(全部用例)                 |
| 异常传参     | sample 无参数校验路径，异常形态不适用  | 不适用                               |
| 混合设备类型 | 契约在 CPU 上即可验证，无多设备语义    | 不适用                             |

未覆盖项及原因：
- 未覆盖：total_count=0 的退化采样。此时 categorical 采样仍执行但
  counts 全零，与空张量语义易混淆，且上游未承诺该形态，不做伪断言。
"""
import os
import unittest

import torch

from torch.distributions.multinomial import Multinomial

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
class TestMultinomialSample(TestCase):
    """Test cases for Multinomial.sample (shape/sum/support contracts)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_sample_default_shape_and_counts_sum(self):
        """Default sample: shape (K,), nonneg integer counts summing to n."""
        torch.manual_seed(0)
        m = Multinomial(10, probs=torch.tensor([0.2, 0.3, 0.5]))
        s = m.sample()
        self.assertEqual(tuple(s.shape), (3,))
        torch.testing.assert_close(s.sum(-1), torch.tensor(10.0))
        self.assertTrue(bool((s >= 0).all()))
        self.assertTrue(bool((s <= 10).all()))
        self.assertTrue(bool((s == s.floor()).all()))

    def test_sample_with_explicit_sample_shape(self):
        """sample((4,)) prepends the requested shape to the counts."""
        torch.manual_seed(1)
        m = Multinomial(10, probs=torch.tensor([0.2, 0.3, 0.5]))
        s = m.sample((4,))
        self.assertEqual(tuple(s.shape), (4, 3))
        torch.testing.assert_close(s.sum(-1), torch.full((4,), 10.0))
        self.assertTrue(bool((s >= 0).all()))

    def test_sample_batched_params(self):
        """Batched probs produce (sample_shape, batch, event) counts."""
        torch.manual_seed(2)
        m = Multinomial(
            6, probs=torch.tensor([[0.1, 0.2, 0.7], [0.5, 0.25, 0.25]]))
        s = m.sample((5,))
        self.assertEqual(tuple(s.shape), (5, 2, 3))
        torch.testing.assert_close(s.sum(-1), torch.full((5, 2), 6.0))
        self.assertTrue(bool(m.support.check(s).all()))

    def test_sample_logits_path(self):
        """Sampling works through the logits parameterization too."""
        torch.manual_seed(3)
        m = Multinomial(8, logits=torch.tensor([0.5, -0.5, 0.0]))
        s = m.sample((3,))
        self.assertEqual(tuple(s.shape), (3, 3))
        torch.testing.assert_close(s.sum(-1), torch.full((3,), 8.0))
        self.assertTrue(bool((s == s.floor()).all()))

    def test_sample_dtype_matches_probs(self):
        """Counts are cast back to the dtype of the probs parameter."""
        torch.manual_seed(4)
        m = Multinomial(5, probs=torch.tensor([0.4, 0.6], dtype=torch.float64))
        s = m.sample()
        self.assertEqual(s.dtype, torch.float64)
        torch.testing.assert_close(s.sum(), torch.tensor(5.0, dtype=torch.float64))

    def test_sample_reproducible_under_fixed_seed(self):
        """Two seeded calls produce identical counts."""
        torch.manual_seed(123)
        m = Multinomial(10, probs=torch.tensor([0.2, 0.3, 0.5]))
        first = m.sample((2,))
        torch.manual_seed(123)
        second = m.sample((2,))
        torch.testing.assert_close(first, second)


if __name__ == "__main__":
    run_tests()
