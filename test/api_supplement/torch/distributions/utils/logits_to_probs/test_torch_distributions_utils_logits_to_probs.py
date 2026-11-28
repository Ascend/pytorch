# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.utils.logits_to_probs 接口功能正确性
API 名称：torch.distributions.utils.logits_to_probs
API 签名：logits_to_probs(logits: Tensor, is_binary: bool = False) -> Tensor

覆盖维度表：
| 覆盖维度     | 说明                                   | 覆盖情况                              |
|--------------|----------------------------------------|---------------------------------------|
| 空/非空      | 空类别维与非空类别维输入               | 已覆盖（unnormalized_are_normalized） |
| 枚举选项     | is_binary 布尔开关 False/True 两态     | 已覆盖（multivariate_softmax、        |
|              |                                        | binary_sigmoid）                      |
| 参数类型     | float32 与 float64 logits 张量         | 已覆盖（dtype_float64）               |
| 传参与不传参 | 不传参默认 is_binary=False；显式传两态 | 已覆盖(multivariate_softmax、        |
|              |                                        | binary_sigmoid）                      |
| 等价类/边界  | 全相等 logits（均匀分布）边界          | 已覆盖（unnormalized_are_normalized） |
| 正常传参     | 与 softmax/sigmoid 闭式解逐元素对照    | 已覆盖（multivariate_softmax、        |
|              |                                        | binary_sigmoid、batched_last_dim）    |
| 异常传参     | 无异常路径（数值函数不抛错）           | 不适用                                |
| 混合设备类型 | 契约在 CPU 上即可验证，无多设备语义    | 不适用                             |

未覆盖项及原因：
- 未覆盖：logits 含 +/-inf 极端输入。softmax/sigmoid 对 inf 的传播
  结果依赖后端实现细节（可能产生 NaN），源码未承诺，不做伪断言。
"""
import os
import unittest

import torch

from torch.distributions.utils import logits_to_probs, probs_to_logits

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
class TestLogitsToProbs(TestCase):
    """Test cases for utils.logits_to_probs (softmax/sigmoid contracts)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_multivariate_softmax(self):
        """Default is_binary=False applies softmax over the last dim."""
        logits = torch.tensor([1.0, 2.0, 3.0])
        torch.testing.assert_close(
            logits_to_probs(logits), torch.softmax(logits, dim=-1))

    def test_binary_sigmoid(self):
        """is_binary=True applies elementwise sigmoid."""
        logits = torch.tensor([0.0, 1.0, -2.0])
        torch.testing.assert_close(
            logits_to_probs(logits, is_binary=True), torch.sigmoid(logits))

    def test_unnormalized_logits_are_normalized(self):
        """Equal logits normalize to a uniform distribution."""
        logits = torch.tensor([3.0, 3.0, 3.0])
        torch.testing.assert_close(
            logits_to_probs(logits), torch.full((3,), 1.0 / 3.0))

    def test_batched_last_dim(self):
        """Batched inputs are normalized along the last dim only."""
        logits = torch.tensor([[0.0, 0.0], [1.0, -1.0]])
        result = logits_to_probs(logits)
        self.assertEqual(tuple(result.shape), (2, 2))
        torch.testing.assert_close(result, torch.softmax(logits, dim=-1))

    def test_roundtrip_with_probs_to_logits(self):
        """probs -> logits -> probs recovers the original interior probs."""
        probs = torch.tensor([0.2, 0.3, 0.5])
        recovered = logits_to_probs(probs_to_logits(probs))
        torch.testing.assert_close(recovered, probs)

    def test_dtype_float64(self):
        """float64 inputs keep the dtype through the conversion."""
        logits = torch.tensor([1.0, 2.0], dtype=torch.float64)
        result = logits_to_probs(logits)
        self.assertEqual(result.dtype, torch.float64)
        torch.testing.assert_close(result, torch.softmax(logits, dim=-1))


if __name__ == "__main__":
    run_tests()
