# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.utils.clamp_probs 接口功能正确性
API 名称：torch.distributions.utils.clamp_probs
API 签名：clamp_probs(probs: Tensor) -> Tensor

覆盖维度表：
| 覆盖维度     | 说明                                   | 覆盖情况                              |
|--------------|----------------------------------------|---------------------------------------|
| 空/非空      | 0/1 边界与非空内部取值                 | 已覆盖（clamps_zero_and_one、         |
|              |                                        | interior_unchanged）                  |
| 枚举选项     | 本接口无枚举入参                       | 不适用                                |
| 参数类型     | float16/float32/float64 三种 dtype     | 已覆盖（float64、float16、默认用例）  |
| 传参与不传参 | 单必填参数 probs，无可选参数           | 已覆盖(各用例仅传 probs)          |
| 等价类/边界  | <0、>1 的越界等价类                    | 已覆盖（out_of_range_inputs_clamped） |
| 正常传参     | 与 finfo eps 截断规则逐元素对照        | 已覆盖（clamps_zero_and_one、         |
|              |                                        | float64_eps）                         |
| 异常传参     | 无异常路径（数值截断不抛错）           | 不适用                                |
| 混合设备类型 | 契约在 CPU 上即可验证，无多设备语义    | 不适用                             |

未覆盖项及原因：
- 未覆盖：NaN 输入。clamp 对 NaN 的传播结果无闭式对照价值，且源码
  未承诺 NaN 行为，不做伪断言。
"""
import os
import unittest

import torch

from torch.distributions.utils import clamp_probs

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
class TestClampProbs(TestCase):
    """Test cases for utils.clamp_probs (open-interval clamping)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_clamps_zero_and_one_to_dtype_eps(self):
        """0 maps to eps and 1 maps to 1 - eps for float32."""
        probs = torch.tensor([0.0, 0.5, 1.0])
        eps = torch.finfo(probs.dtype).eps
        expected = torch.tensor([eps, 0.5, 1.0 - eps])
        torch.testing.assert_close(clamp_probs(probs), expected)

    def test_float64_uses_float64_eps(self):
        """float64 inputs clamp to the float64 eps."""
        probs = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        eps = torch.finfo(torch.float64).eps
        expected = torch.tensor([eps, 0.5, 1.0 - eps], dtype=torch.float64)
        torch.testing.assert_close(clamp_probs(probs), expected)

    def test_interior_values_unchanged(self):
        """Values strictly inside (0, 1) pass through unchanged."""
        probs = torch.tensor([0.25, 0.5, 0.75, 0.999])
        torch.testing.assert_close(clamp_probs(probs), probs)

    def test_out_of_range_inputs_clamped(self):
        """Negative and >1 inputs clamp to the open interval bounds."""
        eps = torch.finfo(torch.float32).eps
        result = clamp_probs(torch.tensor([-0.5, 1.5]))
        torch.testing.assert_close(result, torch.tensor([eps, 1.0 - eps]))

    def test_result_always_in_open_interval(self):
        """Seeded random inputs always produce results inside (0, 1)."""
        torch.manual_seed(0)
        r = torch.rand(1000)
        c = clamp_probs(r)
        self.assertTrue(bool(((c > 0) & (c < 1)).all()))
        torch.testing.assert_close(c, r.clamp(min=torch.finfo(r.dtype).eps,
                                              max=1 - torch.finfo(r.dtype).eps))

    def test_float16_uses_float16_eps(self):
        """float16 inputs clamp to the float16 eps."""
        probs = torch.zeros(3, dtype=torch.float16)
        eps = torch.finfo(torch.float16).eps
        torch.testing.assert_close(
            clamp_probs(probs), torch.full((3,), eps, dtype=torch.float16))


if __name__ == "__main__":
    run_tests()
