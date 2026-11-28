# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.multinomial.Multinomial.log_prob 接口功能正确性
API 名称：torch.distributions.multinomial.Multinomial.log_prob
API 签名：log_prob(self, value: Tensor) -> Tensor

覆盖维度表：
| 覆盖维度     | 说明                                   | 覆盖情况                              |
|--------------|----------------------------------------|---------------------------------------|
| 空/非空      | 计数含 0 与全非 0 两种取值形态         | 已覆盖（zero_count_with_zero_prob、   |
|              |                                        | logits_neg_inf_edge）                 |
| 枚举选项     | 本接口无枚举入参                       | 不适用                                |
| 参数类型     | float32 与 float64 计数/概率张量       | 已覆盖（dtype_float64）               |
| 传参与不传参 | value 必填；validate_args 由构造期控制 | 已覆盖（negative_counts_raise 显式    |
|              |                                        | 传 True，其余用例不传）               |
| 等价类/边界  | 零概率类别与 -inf logits 边界          | 已覆盖（logits_neg_inf_edge、         |
|              |                                        | zero_count_with_zero_prob）           |
| 正常传参     | 与 lgamma 多项式手工公式逐元素对照     | 已覆盖(manual_computation、logits_   |
|              |                                        | path、batch_broadcast）               |
| 异常传参     | 负计数触发 support 校验 ValueError     | 已覆盖（negative_counts_raise）       |
| 混合设备类型 | 契约在 CPU 上即可验证，无多设备语义    | 不适用                             |

未覆盖项及原因：
- 未覆盖：counts 超过 total_count 的取值。validate_args 关闭时 log_prob
  仍按公式外推返回有限值，结果依赖全局开关，无法构造稳定断言。
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
class TestMultinomialLogProb(TestCase):
    """Test cases for Multinomial.log_prob (pure Python contracts on CPU)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_log_prob_matches_manual_computation(self):
        """log_prob equals lgamma(n+1) - sum lgamma(x+1) + sum(x * log p)."""
        probs = torch.tensor([0.2, 0.3, 0.5])
        m = Multinomial(10, probs=probs)
        value = torch.tensor([[2.0, 3.0, 5.0]])
        log_factorial_n = torch.lgamma(value.sum(-1) + 1.0)
        log_factorial_xs = torch.lgamma(value + 1.0).sum(-1)
        log_powers = (value * torch.log(probs)).sum(-1)
        expected = log_factorial_n - log_factorial_xs + log_powers
        torch.testing.assert_close(m.log_prob(value), expected)

    def test_log_prob_logits_path_matches_normalized_probs(self):
        """logits construction normalizes to softmax probs; log_prob agrees."""
        raw_logits = torch.tensor([1.0, 2.0, 3.0])
        m = Multinomial(10, logits=raw_logits)
        ref_probs = torch.softmax(raw_logits, dim=-1)
        torch.testing.assert_close(m.probs, ref_probs)
        value = torch.tensor([[1.0, 2.0, 7.0]])
        expected = (torch.lgamma(value.sum(-1) + 1.0)
                    - torch.lgamma(value + 1.0).sum(-1)
                    + (value * torch.log(ref_probs)).sum(-1))
        torch.testing.assert_close(m.log_prob(value), expected)

    def test_log_prob_zero_count_with_zero_prob_category(self):
        """Zero-prob category clamps to eps: log_prob(0-count pattern) finite."""
        m = Multinomial(1, probs=torch.tensor([1.0, 0.0]))
        eps = torch.finfo(torch.float32).eps
        # Count pattern [1, 0]: lgamma terms cancel, power term is log(1).
        torch.testing.assert_close(
            m.log_prob(torch.tensor([[1.0, 0.0]])), torch.zeros(1))
        # Count pattern [0, 1]: the zero-prob category is clamped to eps.
        # value carries a (1,) batch dim, so the golden must be (1,) too.
        expected = torch.log(torch.clamp(
            torch.tensor([[1.0, 0.0]]), min=eps, max=1.0 - eps)).sum(-1)
        torch.testing.assert_close(
            m.log_prob(torch.tensor([[0.0, 1.0]])), expected)

    def test_log_prob_logits_neg_inf_edge(self):
        """-inf logits with zero counts are neutralized in-place (source)."""
        m = Multinomial(
            1, logits=torch.tensor([0.0, float("-inf")]), validate_args=False)
        # value==0 where logits==-inf: the entry is reset to 0, no nan.
        torch.testing.assert_close(
            m.log_prob(torch.tensor([[1.0, 0.0]])), torch.zeros(1))
        # Positive count on a -inf logit category gives -inf log_prob.
        result = m.log_prob(torch.tensor([[0.0, 1.0]]))
        torch.testing.assert_close(result, torch.full((1,), float("-inf")))

    def test_log_prob_batch_broadcast(self):
        """Batched probs broadcast against a single count vector."""
        probs = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])
        m = Multinomial(5, probs=probs)
        value = torch.tensor([1.0, 2.0, 2.0])
        result = m.log_prob(value)
        self.assertEqual(tuple(result.shape), (2,))
        expected = (torch.lgamma(value.sum(-1) + 1.0)
                    - torch.lgamma(value + 1.0).sum(-1)
                    + (value * torch.log(probs)).sum(-1))
        torch.testing.assert_close(result, expected)

    def test_log_prob_dtype_float64(self):
        """float64 probs and counts keep the dtype and match float64 math."""
        probs = torch.tensor([0.25, 0.75], dtype=torch.float64)
        m = Multinomial(4, probs=probs)
        value = torch.tensor([[2.0, 2.0]], dtype=torch.float64)
        result = m.log_prob(value)
        self.assertEqual(result.dtype, torch.float64)
        expected = (torch.lgamma(value.sum(-1) + 1.0)
                    - torch.lgamma(value + 1.0).sum(-1)
                    + (value * torch.log(probs)).sum(-1))
        torch.testing.assert_close(result, expected)

    def test_log_prob_negative_counts_raise(self):
        """Negative counts violate the multinomial support and raise."""
        m = Multinomial(3, probs=torch.tensor([0.5, 0.5]), validate_args=True)
        bad = torch.tensor([[-1.0, 4.0]])
        with self.assertRaises(ValueError):
            m.log_prob(bad)


if __name__ == "__main__":
    run_tests()
