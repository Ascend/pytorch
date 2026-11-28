# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.multinomial.Multinomial.logits 接口功能正确性
API 名称：torch.distributions.multinomial.Multinomial.logits
API 签名：logits(self) -> Tensor（property）

覆盖维度表：
| 覆盖维度     | 说明                                   | 覆盖情况                              |
|--------------|----------------------------------------|---------------------------------------|
| 空/非空      | 概率含 0(边界)与全为正两种形态       | 已覆盖(zero_prob_category_clamped、  |
|              |                                        | log_normalized_probs）                |
| 枚举选项     | 本接口无枚举入参                       | 不适用                                |
| 参数类型     | float32 与 float64 参数张量            | 已覆盖（dtype_float64）               |
| 传参与不传参 | 无入参 property；probs/logits 二选一   | 已覆盖（log_normalized_probs 用 probs，|
|              | 构造形态                               | normalized_by_logsumexp 用 logits）   |
| 等价类/边界  | 零概率类别经 clamp_probs 的 eps 边界   | 已覆盖（zero_prob_category_clamped）  |
| 正常传参     | 与 log 归一化 / logsumexp 归一化对照   | 已覆盖（log_normalized_probs、        |
|              |                                        | normalized_by_logsumexp、batched）    |
| 异常传参     | 无异常路径(无入参不触发参数校验)     | 不适用                                |
| 混合设备类型 | 契约在 CPU 上即可验证，无多设备语义    | 不适用                             |

未覆盖项及原因：
- 未覆盖：logits 含 NaN 的构造输入。约束检查对 NaN 的行为依赖全局
  validate_args 开关，且归一化结果全为 NaN 无对照价值，不做伪断言。
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
class TestMultinomialLogits(TestCase):
    """Test cases for Multinomial.logits (pure Python contracts on CPU)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_logits_equals_log_of_normalized_probs(self):
        """logits == log(probs / sum(probs)) along the last dim."""
        raw = torch.tensor([2.0, 3.0, 5.0])
        m = Multinomial(10, probs=raw)
        normalized = raw / raw.sum(-1, keepdim=True)
        torch.testing.assert_close(m.probs, normalized)
        torch.testing.assert_close(m.logits, torch.log(normalized))

    def test_logits_normalized_by_logsumexp(self):
        """Raw logits are shifted by logsumexp and match log_softmax."""
        raw = torch.tensor([1.0, 2.0, 3.0])
        m = Multinomial(1, logits=raw)
        expected = raw - raw.logsumexp(dim=-1, keepdim=True)
        torch.testing.assert_close(m.logits, expected)
        torch.testing.assert_close(m.logits, torch.log_softmax(raw, dim=-1))

    def test_logits_batched_rowwise_normalization(self):
        """Batched probs are normalized row by row."""
        raw = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])
        m = Multinomial(1, probs=raw)
        result = m.logits
        self.assertEqual(tuple(result.shape), (2, 3))
        normalized = raw / raw.sum(-1, keepdim=True)
        torch.testing.assert_close(result, torch.log(normalized))

    def test_logits_zero_prob_category_clamped(self):
        """A zero-prob category is clamped to eps before taking the log."""
        m = Multinomial(1, probs=torch.tensor([1.0, 0.0]))
        eps = torch.finfo(torch.float32).eps
        clamped = torch.clamp(
            torch.tensor([1.0, 0.0]), min=eps, max=1.0 - eps)
        torch.testing.assert_close(m.logits, torch.log(clamped))

    def test_logits_dtype_float64(self):
        """float64 params keep the dtype in the returned logits."""
        raw = torch.tensor([1.0, 2.0], dtype=torch.float64)
        m = Multinomial(1, logits=raw)
        self.assertEqual(m.logits.dtype, torch.float64)
        torch.testing.assert_close(
            m.logits, raw - raw.logsumexp(dim=-1, keepdim=True))

    def test_logits_exponentiate_to_simplex(self):
        """exp(logits) sums to 1, matching the documented normalization."""
        m = Multinomial(1, logits=torch.tensor([0.5, -0.5, 2.0]))
        torch.testing.assert_close(m.logits.exp().sum(-1), torch.tensor(1.0))


if __name__ == "__main__":
    run_tests()
