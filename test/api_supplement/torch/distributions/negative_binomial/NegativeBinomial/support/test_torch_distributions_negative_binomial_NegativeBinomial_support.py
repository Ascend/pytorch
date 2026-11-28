# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.negative_binomial.NegativeBinomial.support
接口功能正确性
API 名称：torch.distributions.negative_binomial.NegativeBinomial.support
API 签名：support = constraints.nonnegative_integer（类属性）

覆盖维度表：
| 覆盖维度     | 说明                                   | 覆盖情况                              |
|--------------|----------------------------------------|---------------------------------------|
| 空/非空      | 支撑集上样本(非空)与空约束形态       | 已覆盖(check_on_samples)            |
| 枚举选项     | 本接口无枚举入参                       | 不适用                                |
| 参数类型     | float32 与 float64 参数张量            | 已覆盖（dtype_float64）               |
| 传参与不传参 | 无入参类属性；probs/logits 二选一构造  | 已覆盖(is_nonnegative_integer、      |
|              |                                        | logits_path）                         |
| 等价类/边界  | 0/负数/小数三类边界取值                | 已覆盖（check_boundary_values）       |
| 正常传参     | 约束对象同一性与 check 布尔结果断言    | 已覆盖(is_nonnegative_integer 等)   |
| 异常传参     | check 返回 False 而非抛错，无异常路径  | 不适用                                |
| 混合设备类型 | 契约在 CPU 上即可验证，无多设备语义    | 不适用                             |

未覆盖项及原因：
- 未覆盖：total_count=0 的退化分布支撑集。源码仅对 _gamma 构造关闭
  校验，support 本身与 total_count 无关，无独立可断言的差异行为。
"""
import os
import unittest

import torch

from torch.distributions import constraints
from torch.distributions.negative_binomial import NegativeBinomial

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
class TestNegativeBinomialSupport(TestCase):
    """Test cases for NegativeBinomial.support (constraint contracts)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_support_is_nonnegative_integer_constraint(self):
        """support is the shared nonnegative_integer constraint object."""
        nb = NegativeBinomial(2.0, probs=torch.tensor([0.5]))
        self.assertIs(nb.support, constraints.nonnegative_integer)
        self.assertIs(NegativeBinomial.support, constraints.nonnegative_integer)

    def test_support_check_boundary_values(self):
        """0 and positive integers pass; negative and fractional fail."""
        nb = NegativeBinomial(2.0, probs=torch.tensor([0.5]))
        self.assertTrue(bool(nb.support.check(
            torch.tensor([0.0, 1.0, 7.0])).all()))
        self.assertFalse(bool(nb.support.check(torch.tensor([-1.0])).any()))
        self.assertFalse(bool(nb.support.check(torch.tensor([0.5])).any()))

    def test_support_check_on_samples(self):
        """All seeded samples lie in the nonnegative-integer support."""
        torch.manual_seed(0)
        nb = NegativeBinomial(torch.tensor([2.0]), probs=torch.tensor([0.3]))
        s = nb.sample((100,))
        self.assertEqual(tuple(s.shape), (100, 1))
        self.assertTrue(bool(nb.support.check(s).all()))
        torch.testing.assert_close(s, s.floor())

    def test_support_logits_path(self):
        """The logits parameterization exposes the same support."""
        nb = NegativeBinomial(2.0, logits=torch.tensor([0.25]))
        self.assertIs(nb.support, constraints.nonnegative_integer)
        self.assertTrue(bool(nb.support.check(torch.tensor([3.0])).all()))
        self.assertFalse(bool(nb.support.check(torch.tensor([1.5])).any()))

    def test_support_batched_params_share_constraint(self):
        """Batched total_count/probs still share the class-level support."""
        nb = NegativeBinomial(
            torch.tensor([2.0, 5.0]), probs=torch.tensor([0.3, 0.7]))
        self.assertIs(nb.support, constraints.nonnegative_integer)
        torch.manual_seed(1)
        s = nb.sample((10,))
        self.assertEqual(tuple(s.shape), (10, 2))
        self.assertTrue(bool(nb.support.check(s).all()))

    def test_support_check_dtype_float64(self):
        """check works on float64 inputs without dtype complaints."""
        nb = NegativeBinomial(
            2.0, probs=torch.tensor([0.5], dtype=torch.float64))
        self.assertTrue(bool(nb.support.check(
            torch.tensor([0.0, 4.0], dtype=torch.float64)).all()))
        self.assertFalse(bool(nb.support.check(
            torch.tensor([-0.25], dtype=torch.float64)).any()))


if __name__ == "__main__":
    run_tests()
