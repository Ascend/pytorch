# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.relaxed_bernoulli.RelaxedBernoulli.expand
接口功能正确性
API 名称：torch.distributions.relaxed_bernoulli.RelaxedBernoulli.expand
API 签名：expand(self, batch_shape, _instance=None) -> RelaxedBernoulli

覆盖维度表：
| 覆盖维度     | 说明                                   | 覆盖情况                              |
|--------------|----------------------------------------|---------------------------------------|
| 空/非空      | (1,) 空批次扩展为多元素批次            | 已覆盖（batch_shape_and_params 等）   |
| 枚举选项     | 本接口无枚举入参                       | 不适用                                |
| 参数类型     | float32 参数张量在目标设备构造并全链路 | 已覆盖（batch_shape_and_params、      |
|              | 执行；float64 闭式解契约在 CPU 钉死    | log_prob_closed_form）                |
| 传参与不传参 | batch_shape 必填；_instance 走默认路径 | 已覆盖(全部用例均不传 _instance)   |
| 等价类/边界  | probs 接近 1 的边界取值参与扩展        | 已覆盖（batch_shape_and_params）      |
| 正常传参     | 扩展后 log_prob 与闭式解逐元素对照     | 已覆盖（log_prob_closed_form、        |
|              |                                        | logits_path_consistency）             |
| 异常传参     | 与原 batch 不兼容的形状应报错          | 已覆盖（incompatible_batch_shape）    |
| 混合设备类型 | 无跨设备混用输入场景；float32 全链路   | 不适用（见参数类型行）                |
|              | 在目标设备执行，float64 在 CPU 验证    |                                       |

未覆盖项及原因：
- 未覆盖：显式传入 _instance 的复用路径。expand 仅委托基类实现，
  外部实例复用的类型校验未在源码中稳定承诺，不做伪断言。
- 未覆盖：float64 参数的目标设备执行路径。NPU 对 float64 elementwise
  kernel 的支持不保证，闭式解 golden 在 CPU 上钉死契约；目标设备路径由
  float32 用例覆盖。
"""
import os
import unittest

import torch

from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

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
    # TestCase. Floating point comparisons use torch.testing.assert_close,
    # which is device-agnostic and works with the fallback as well.
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

# transfer_to_npu rebinds torch.cuda.* and torch.Event to the NPU
# implementations, so the test body uses generic entry points unmodified
# on NPU and CUDA. Keep AFTER the testcase import above.
if DEVICE_NAME == 'npu':
    import torch_npu.contrib.transfer_to_npu  # noqa: F401


def _make_relaxed_bernoulli(dtype=torch.float32, device=None):
    """Build a (1,)-batched RelaxedBernoulli whose batch can expand to (6,)."""
    temp = torch.tensor([2.2], dtype=dtype, device=device)
    probs = torch.tensor([0.99], dtype=dtype, device=device)
    return RelaxedBernoulli(temp, probs=probs)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestRelaxedBernoulliExpand(TestCase):
    """Test cases for RelaxedBernoulli.expand (batch/log_prob contracts)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_expand_batch_shape_and_params(self):
        """Expansion grows the batch shape and broadcasts parameters."""
        m = _make_relaxed_bernoulli(device=self.device)
        e = m.expand((6,))
        self.assertEqual(tuple(m.batch_shape), (1,))
        self.assertEqual(tuple(e.batch_shape), (6,))
        torch.testing.assert_close(e.temperature, m.temperature)
        torch.testing.assert_close(e.probs, m.probs.expand(6))

    def test_expand_log_prob_closed_form(self):
        """Expanded log_prob matches the relaxed-Bernoulli closed form."""
        m = _make_relaxed_bernoulli(torch.float64)
        e = m.expand((6,))
        value = torch.tensor(0.3, dtype=torch.float64)
        probs = torch.tensor([0.99], dtype=torch.float64)
        temp = torch.tensor([2.2], dtype=torch.float64)
        logits = torch.log(probs) - torch.log1p(-probs)
        # RelaxedBernoulli = SigmoidTransform(LogitRelaxedBernoulli): the
        # temperature enters at the logit-space point z = logit(value), and
        # the sigmoid jacobian adds -log(y) - log(1-y).
        z = value.log() - torch.log1p(-value)
        diff = logits - temp * z
        expected = (temp.log() + diff - 2.0 * torch.log1p(diff.exp())
                    - value.log() - torch.log1p(-value))
        result = e.log_prob(value)
        self.assertEqual(tuple(result.shape), (6,))
        torch.testing.assert_close(result, expected.expand(6))

    def test_expand_logits_path_consistency(self):
        """logits-parameterized expansion scores values identically."""
        # float64 golden is pinned on CPU: NPU float64 elementwise kernel
        # support is not guaranteed; the device path is covered by the
        # float32 cases above/below.
        temp = torch.tensor([1.5], dtype=torch.float64)
        logits = torch.tensor([0.0], dtype=torch.float64)
        m = RelaxedBernoulli(temp, logits=logits)
        e = m.expand((3,))
        self.assertEqual(tuple(e.batch_shape), (3,))
        value = torch.tensor(0.7, dtype=torch.float64)
        torch.testing.assert_close(e.log_prob(value), m.log_prob(value).expand(3))

    def test_expand_batched_params(self):
        """A (4,)-batch expands to (2, 4) via leading-dim growth."""
        temp = torch.tensor([2.2], device=self.device)
        probs = torch.tensor([0.1, 0.3, 0.6, 0.99], device=self.device)
        m = RelaxedBernoulli(temp, probs=probs)
        self.assertEqual(tuple(m.batch_shape), (4,))
        e = m.expand((2, 4))
        self.assertEqual(tuple(e.batch_shape), (2, 4))
        value = torch.tensor(0.4, device=self.device)
        torch.testing.assert_close(e.log_prob(value), m.log_prob(value).expand(2, 4))

    def test_expand_rsample_within_unit_interval(self):
        """Expanded rsample stays in (0, 1) and passes support checks."""
        torch.manual_seed(0)
        m = _make_relaxed_bernoulli(device=self.device)
        e = m.expand((6,))
        s = e.rsample((5,))
        self.assertEqual(tuple(s.shape), (5, 6))
        self.assertTrue(bool(((s > 0) & (s < 1)).all()))
        self.assertTrue(bool(e.support.check(s).all()))

    def test_expand_incompatible_batch_shape_raises(self):
        """Expanding (4,) to (2, 5) violates broadcasting rules."""
        m = RelaxedBernoulli(
            torch.tensor([2.2], device=self.device),
            probs=torch.tensor([0.1, 0.3, 0.6, 0.99], device=self.device))
        with self.assertRaises(RuntimeError):
            m.expand((2, 5))


if __name__ == "__main__":
    run_tests()
