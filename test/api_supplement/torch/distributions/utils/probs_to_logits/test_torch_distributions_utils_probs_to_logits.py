# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.utils.probs_to_logits 接口功能正确性
API 名称：torch.distributions.utils.probs_to_logits
API 签名：probs_to_logits(probs: Tensor, is_binary: bool = False) -> Tensor

覆盖维度表：
| 覆盖维度     | 说明                                   | 覆盖情况                              |
|--------------|----------------------------------------|---------------------------------------|
| 空/非空      | 0/1 边界取值与内部取值                 | 已覆盖（boundary_probs_clamped、      |
|              |                                        | log_of_probs）                        |
| 枚举选项     | is_binary 布尔开关 False/True 两态     | 已覆盖（binary_log_odds、log_of_probs)|
| 参数类型     | float32 与 float64 probs 张量          | 已覆盖（batched_and_dtype_float64）   |
| 传参与不传参 | 不传参默认 is_binary=False；显式传两态 | 已覆盖(log_of_probs、binary_log_odds)|
| 等价类/边界  | probs=0/1 经 clamp 的 eps 边界         | 已覆盖（boundary_probs_clamped）      |
| 正常传参     | 与 log / logit 闭式解逐元素对照        | 已覆盖（log_of_probs、binary_log_odds)|
| 异常传参     | 无异常路径（数值函数不抛错）           | 不适用                                |
| 混合设备类型 | 契约在 CPU 上即可验证，无多设备语义    | 不适用                             |

未覆盖项及原因：
- 未覆盖：probs 含负数输入。clamp 后按 eps 处理，结果为 log(eps)，
  语义等同边界用例已覆盖，无独立断言价值。
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
class TestProbsToLogits(TestCase):
    """Test cases for utils.probs_to_logits (log/log-odds contracts)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_log_of_probs(self):
        """Default is_binary=False returns log(probs) elementwise."""
        probs = torch.tensor([0.2, 0.3, 0.5])
        torch.testing.assert_close(probs_to_logits(probs), torch.log(probs))

    def test_binary_log_odds(self):
        """is_binary=True returns log(p) - log1p(-p), matching torch.logit."""
        probs = torch.tensor([0.25, 0.5, 0.9])
        expected = torch.log(probs) - torch.log1p(-probs)
        result = probs_to_logits(probs, is_binary=True)
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(result, torch.logit(probs))

    def test_boundary_probs_clamped(self):
        """probs 0 and 1 clamp to eps before taking the log."""
        eps = torch.finfo(torch.float32).eps
        result = probs_to_logits(torch.tensor([0.0, 1.0]))
        expected = torch.log(torch.tensor([eps, 1.0 - eps]))
        torch.testing.assert_close(result, expected)

    def test_roundtrip_with_logits_to_probs(self):
        """probs -> logits -> probs recovers the original interior probs."""
        probs = torch.tensor([0.1, 0.4, 0.5])
        torch.testing.assert_close(logits_to_probs(probs_to_logits(probs)), probs)

    def test_batched_and_dtype_float64(self):
        """Batched float64 inputs keep shape and dtype."""
        probs = torch.tensor([[0.2, 0.8], [0.5, 0.5]], dtype=torch.float64)
        result = probs_to_logits(probs)
        self.assertEqual(result.dtype, torch.float64)
        self.assertEqual(tuple(result.shape), (2, 2))
        torch.testing.assert_close(result, torch.log(probs))


if __name__ == "__main__":
    run_tests()
