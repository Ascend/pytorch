# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.relaxed_categorical.
RelaxedOneHotCategorical.base_dist 接口功能正确性
API 名称：torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.base_dist
API 签名：base_dist: ExpRelaxedCategorical（实例属性）

覆盖维度表：
| 空/非空      | 空批次(标量批次)与类别维事件形态     | 已覆盖(shapes)                      |
| 枚举选项     | 本接口无枚举入参                       | 不适用                                |
| 参数类型     | float32 与 float64 温度/概率张量       | 已覆盖（dtype_float64）               |
| 传参与不传参 | 无入参属性；probs/logits 二选一构造    | 已覆盖(params_match、logits_path)   |
| 等价类/边界  | 多类别 K=3 与 K=4 两种事件宽度         | 已覆盖（params_match、logits_path）   |
| 正常传参     | 基分布类型、参数与 Exp 复合一致性      | 已覆盖(is_exp_relaxed_categorical、  |
|              |                                        | rsample_through_exp、log_simplex）    |
| 异常传参     | 无异常路径(属性读取不触发校验)       | 不适用                                |
| 混合设备类型 | 契约在 CPU 上即可验证，无多设备语义    | 不适用                             |

未覆盖项及原因：
- 未覆盖：直接替换 base_dist 后的行为。源码未承诺替换语义，替换后
  一致性属于未定义行为，不做伪断言。
"""
import os
import unittest

import torch

from torch.distributions.relaxed_categorical import (
    ExpRelaxedCategorical,
    RelaxedOneHotCategorical,
)
from torch.distributions.transforms import ExpTransform

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
class TestRelaxedOneHotCategoricalBaseDist(TestCase):
    """Test cases for RelaxedOneHotCategorical.base_dist."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_base_dist_is_exp_relaxed_categorical(self):
        """base_dist is an ExpRelaxedCategorical per the source contract."""
        m = RelaxedOneHotCategorical(
            torch.tensor([2.2]), probs=torch.tensor([0.1, 0.2, 0.3, 0.4]))
        self.assertIsInstance(m.base_dist, ExpRelaxedCategorical)

    def test_base_dist_temperature_and_probs_match(self):
        """base_dist carries the exact temperature and probs parameters."""
        temp = torch.tensor([2.2])
        probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
        m = RelaxedOneHotCategorical(temp, probs=probs)
        base = m.base_dist
        torch.testing.assert_close(base.temperature, temp)
        torch.testing.assert_close(base.probs, probs)
        torch.testing.assert_close(m.temperature, temp)

    def test_base_dist_logits_path(self):
        """With logits input, base_dist.logits is the normalized form the
        base Categorical derives from them (log of the clamped softmax) —
        NOT the raw logits; the raw values are recoverable via the probs
        path, and the wrapper .logits matches the base form."""
        logits = torch.tensor([0.0, 1.0, -1.0, 2.0])
        m = RelaxedOneHotCategorical(torch.tensor([1.5]), logits=logits)
        ref = torch.distributions.Categorical(logits=logits)
        torch.testing.assert_close(m.base_dist.logits, ref.logits)
        torch.testing.assert_close(m.base_dist.probs, ref.probs)
        torch.testing.assert_close(m.logits, m.base_dist.logits)

    def test_base_dist_shapes(self):
        """base_dist is scalar-batched with an event of K categories."""
        m = RelaxedOneHotCategorical(
            torch.tensor([2.2]), probs=torch.tensor([0.1, 0.2, 0.3, 0.4]))
        self.assertEqual(tuple(m.base_dist.batch_shape), ())
        self.assertEqual(tuple(m.base_dist.event_shape), (4,))
        self.assertEqual(tuple(m.batch_shape), ())
        self.assertEqual(tuple(m.event_shape), (4,))

    def test_base_dist_rsample_through_exp_equals_rsample(self):
        """wrapper.rsample == ExpTransform()(base_dist.rsample)."""
        torch.manual_seed(0)
        m = RelaxedOneHotCategorical(
            torch.tensor([2.2]), probs=torch.tensor([0.1, 0.2, 0.3, 0.4]))
        s = m.rsample()
        torch.manual_seed(0)
        s_ref = ExpTransform()(m.base_dist.rsample())
        torch.testing.assert_close(s, s_ref)
        torch.testing.assert_close(s.sum(-1), torch.tensor(1.0))

    def test_base_dist_rsample_on_log_simplex(self):
        """base_dist samples are log-simplex points: exp() sums to 1."""
        torch.manual_seed(1)
        m = RelaxedOneHotCategorical(
            torch.tensor([2.2]), probs=torch.tensor([0.2, 0.3, 0.5]))
        b = m.base_dist.rsample((7,))
        self.assertEqual(tuple(b.shape), (7, 3))
        torch.testing.assert_close(b.exp().sum(-1), torch.ones(7))

    def test_base_dist_dtype_float64(self):
        """float64 parameters propagate to the base distribution."""
        temp = torch.tensor([2.2], dtype=torch.float64)
        probs = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
        m = RelaxedOneHotCategorical(temp, probs=probs)
        self.assertEqual(m.base_dist.temperature.dtype, torch.float64)
        torch.testing.assert_close(m.base_dist.probs, probs)


if __name__ == "__main__":
    run_tests()
