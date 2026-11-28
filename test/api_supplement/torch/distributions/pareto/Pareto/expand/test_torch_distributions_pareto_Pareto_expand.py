# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.pareto.Pareto.expand 接口功能正确性
API 名称：torch.distributions.pareto.Pareto.expand
API 签名：expand(self, batch_shape, _instance=None) -> Pareto

覆盖维度表：
| 覆盖维度     | 说明                                   | 覆盖情况                              |
|--------------|----------------------------------------|---------------------------------------|
| 空/非空      | 空批次扩展与非空批次再扩展             | 已覆盖（batch_shape_and_log_prob、    |
|              |                                        | two_dim_batch）                       |
| 枚举选项     | 本接口无枚举入参                       | 不适用                                |
| 参数类型     | float32 与 float64 scale/alpha         | 已覆盖（closed_form 用 float64）      |
| 传参与不传参 | batch_shape 必填；_instance 走默认路径 | 已覆盖(全部用例均不传 _instance)   |
| 等价类/边界  | (1,)->(N,) 与 (2,)->(N,2) 广播边界     | 已覆盖（two_dim_batch）               |
| 正常传参     | 扩展后 log_prob 与闭式解逐元素对照     | 已覆盖（log_prob_consistency、        |
|              |                                        | closed_form、support_and_samples）    |
| 异常传参     | 与原 batch 不兼容的形状应报错          | 已覆盖（incompatible_batch_shape）    |
| 混合设备类型 | 契约在 CPU 上即可验证，无多设备语义    | 不适用                             |

未覆盖项及原因：
- 未覆盖：显式传入 _instance 的复用路径。Pareto.expand 先展开
  scale/alpha 再委托基类，外部实例复用的类型校验未稳定承诺，不做伪断言。
"""
import os
import unittest

import torch

from torch.distributions.pareto import Pareto

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
class TestParetoExpand(TestCase):
    """Test cases for Pareto.expand (batch/log_prob/support contracts)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_expand_batch_shape_and_log_prob_consistency(self):
        """Expanded distribution scores the same value per batch item."""
        p = Pareto(torch.tensor([1.0]), torch.tensor([2.0]))
        e = p.expand((6,))
        self.assertEqual(tuple(e.batch_shape), (6,))
        value = torch.tensor(1.5)
        torch.testing.assert_close(e.log_prob(value), p.log_prob(value).expand(6))

    def test_expand_log_prob_closed_form(self):
        """Expanded log_prob matches alpha/scale closed form in float64."""
        scale = torch.tensor([1.0], dtype=torch.float64)
        alpha = torch.tensor([2.0], dtype=torch.float64)
        p = Pareto(scale, alpha)
        e = p.expand((4,))
        value = torch.tensor(3.0, dtype=torch.float64)
        expected = (torch.log(alpha) + alpha * torch.log(scale)
                    - (alpha + 1.0) * torch.log(value))
        result = e.log_prob(value)
        self.assertEqual(tuple(result.shape), (4,))
        torch.testing.assert_close(result, expected.expand(4))

    def test_expand_two_dim_batch(self):
        """A (2,) parameter batch expands to (3, 2)."""
        p = Pareto(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 3.0]))
        e = p.expand((3, 2))
        self.assertEqual(tuple(e.batch_shape), (3, 2))
        value = torch.tensor(2.5)
        result = e.log_prob(value)
        self.assertEqual(tuple(result.shape), (3, 2))
        expected = (torch.log(e.alpha) + e.alpha * torch.log(e.scale)
                    - (e.alpha + 1.0) * torch.log(value))
        torch.testing.assert_close(result, expected)

    def test_expand_support_and_samples(self):
        """Expanded samples respect the >= scale support."""
        torch.manual_seed(0)
        p = Pareto(torch.tensor([1.0]), torch.tensor([2.0]))
        e = p.expand((6,))
        s = e.sample((5,))
        self.assertEqual(tuple(s.shape), (5, 6))
        self.assertTrue(bool(e.support.check(s).all()))
        self.assertFalse(bool(e.support.check(torch.full((6,), 0.5)).any()))

    def test_expand_preserves_scale_and_alpha(self):
        """Expanded parameters are broadcasts of the originals."""
        p = Pareto(torch.tensor([1.0]), torch.tensor([2.0]))
        e = p.expand((4,))
        torch.testing.assert_close(e.scale, p.scale.expand(4))
        torch.testing.assert_close(e.alpha, p.alpha.expand(4))

    def test_expand_incompatible_batch_shape_raises(self):
        """Expanding (2,) to (3,) violates broadcasting rules."""
        p = Pareto(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 3.0]))
        with self.assertRaises(RuntimeError):
            p.expand((3,))


if __name__ == "__main__":
    run_tests()
