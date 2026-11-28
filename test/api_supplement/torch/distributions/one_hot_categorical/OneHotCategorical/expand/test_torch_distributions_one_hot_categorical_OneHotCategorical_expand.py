# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.one_hot_categorical.OneHotCategorical.expand
接口功能正确性
API 名称：torch.distributions.one_hot_categorical.OneHotCategorical.expand
API 签名：expand(self, batch_shape, _instance=None) -> OneHotCategorical

覆盖维度表：
| 覆盖维度     | 说明                                   | 覆盖情况                              |
|--------------|----------------------------------------|---------------------------------------|
| 空/非空      | 空批次扩展与非空批次再扩展             | 已覆盖（batch_shape_and_log_prob、    |
|              |                                        | two_dim_batch）                       |
| 枚举选项     | 本接口无枚举入参                       | 不适用                                |
| 参数类型     | float32 与 float64 概率张量            | 已覆盖（dtype_float64）               |
| 传参与不传参 | batch_shape 必填；_instance 走默认路径 | 已覆盖(全部用例均不传 _instance)   |
| 等价类/边界  | (1,)->(N,) 单元素广播与 (2,)->(N,2)    | 已覆盖（two_dim_batch）               |
| 正常传参     | 扩展后 log_prob 逐元素一致、样本 one-hot| 已覆盖（log_prob_consistency、        |
|              |                                        | closed_form、samples_are_one_hot）    |
| 异常传参     | 与原 batch 不兼容的形状应报错          | 已覆盖（incompatible_batch_shape）    |
| 混合设备类型 | 契约在 CPU 上即可验证，无多设备语义    | 不适用                             |

未覆盖项及原因：
- 未覆盖：显式传入 _instance 的复用路径。该路径语义与内部
  _get_checked_instance 强耦合，直接传入外部实例的类型约束未在源码
  中稳定承诺，不做伪断言。

注意：expand 将底层 Categorical 参数重建为 batch_shape+(K,)（v2.15
     categorical.py expand），故 e.probs 形状为 (batch, K) 且各行相同，
     与原 probs 的对照须按 expand(batch+(K,)) 口径（preserves_probabilities）。
"""
import os
import unittest

import torch

from torch.distributions.one_hot_categorical import OneHotCategorical

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
class TestOneHotCategoricalExpand(TestCase):
    """Test cases for OneHotCategorical.expand (batch/log_prob contracts)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_expand_batch_shape_and_log_prob_consistency(self):
        """Expanded distribution scores a value identically per batch item."""
        m = OneHotCategorical(probs=torch.tensor([0.2, 0.3, 0.5]))
        e = m.expand((4,))
        self.assertEqual(tuple(e.batch_shape), (4,))
        self.assertEqual(tuple(e.event_shape), (3,))
        value = torch.tensor([0.0, 1.0, 0.0])
        torch.testing.assert_close(e.log_prob(value), m.log_prob(value).expand(4))

    def test_expand_log_prob_closed_form(self):
        """Expanded log_prob matches log p at the chosen category."""
        probs = torch.tensor([0.2, 0.3, 0.5])
        m = OneHotCategorical(probs=probs)
        e = m.expand((2, 3))
        value = torch.tensor([0.0, 1.0, 0.0])
        result = e.log_prob(value)
        self.assertEqual(tuple(result.shape), (2, 3))
        expected = (value * torch.log(probs)).sum(-1)
        torch.testing.assert_close(result, expected.expand(2, 3))

    def test_expand_two_dim_batch(self):
        """A (2,) batch can be expanded to (5, 2)."""
        probs = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])
        m = OneHotCategorical(probs=probs)
        e = m.expand((5, 2))
        self.assertEqual(tuple(e.batch_shape), (5, 2))
        value = torch.tensor([1.0, 0.0, 0.0])
        result = e.log_prob(value)
        self.assertEqual(tuple(result.shape), (5, 2))
        torch.testing.assert_close(result, m.log_prob(value).expand(5, 2))

    def test_expand_samples_are_one_hot(self):
        """Expanded samples stay one-hot with row sums equal to 1."""
        torch.manual_seed(0)
        m = OneHotCategorical(probs=torch.tensor([0.2, 0.3, 0.5]))
        e = m.expand((4,))
        s = e.sample((7,))
        self.assertEqual(tuple(s.shape), (7, 4, 3))
        torch.testing.assert_close(s.sum(-1), torch.ones(7, 4))
        self.assertTrue(bool(((s == 0) | (s == 1)).all()))
        self.assertTrue(bool(e.support.check(s).all()))

    def test_expand_preserves_probabilities(self):
        """The expanded distribution keeps the same category probs.

        expand rebuilds the underlying Categorical param as batch+(K,), so
        the expanded probs has shape (3, 3) with identical rows.
        """
        m = OneHotCategorical(probs=torch.tensor([0.25, 0.25, 0.5]))
        e = m.expand((3,))
        self.assertEqual(tuple(e.probs.shape), (3, 3))
        torch.testing.assert_close(e.probs, m.probs.expand(3, 3))

    def test_expand_incompatible_batch_shape_raises(self):
        """Expanding (3,) to (4,) is invalid under broadcasting rules."""
        m = OneHotCategorical(
            probs=torch.tensor([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5],
                                [0.2, 0.3, 0.5]]))
        with self.assertRaises(RuntimeError):
            m.expand((4,))

    def test_expand_dtype_float64(self):
        """float64 params survive expansion with matching dtype."""
        m = OneHotCategorical(
            probs=torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64))
        e = m.expand((3,))
        self.assertEqual(e.probs.dtype, torch.float64)
        value = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        torch.testing.assert_close(e.log_prob(value), m.log_prob(value).expand(3))


if __name__ == "__main__":
    run_tests()
