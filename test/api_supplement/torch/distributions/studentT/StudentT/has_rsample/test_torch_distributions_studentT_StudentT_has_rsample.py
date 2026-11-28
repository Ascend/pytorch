# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.studentT.StudentT.has_rsample 接口功能正确性
API 名称：torch.distributions.studentT.StudentT.has_rsample
API 签名：has_rsample = True（类属性）

覆盖维度表：
| 覆盖维度     | 说明                                   | 覆盖情况                              |
|--------------|----------------------------------------|---------------------------------------|
| 空/非空      | 空参数批次与标量参数批次实例           | 已覆盖(is_true、multiple_params)    |
| 枚举选项     | 布尔类属性仅有 True/False 两态         | 已覆盖(is_true 断言恒为 True)       |
| 参数类型     | df/loc/scale 以张量构造的实例          | 已覆盖（multiple_params）             |
| 传参与不传参 | 类属性无入参；实例化入参不影响属性   | 已覆盖(is_true、multiple_params)  |
| 等价类/边界  | 大自由度 df=5 与 df 张量批量           | 已覆盖（rsample_available）           |
| 正常传参     | 属性恒为 True 且 rsample 真实可用      | 已覆盖（rsample_available）           |
| 异常传参     | 无异常路径(常量属性读取不触发校验)   | 不适用                                |
| 混合设备类型 | 契约在 CPU 上即可验证，无多设备语义    | 不适用                             |

未覆盖项及原因：
- 未覆盖：rsample 对 df 的梯度数值正确性。梯度经 _standard_gamma 隐
  含重参数化实现，与采样 RNG 路径耦合，本文件仅断言 requires_grad，
  不复算梯度值以免伪对照。
"""
import os
import unittest

import torch

from torch.distributions.studentT import StudentT

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
class TestStudentTHasRsample(TestCase):
    """Test cases for StudentT.has_rsample (class attribute contract)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_has_rsample_is_true(self):
        """The class attribute and the instance attribute are both True."""
        self.assertIs(StudentT.has_rsample, True)
        m = StudentT(torch.tensor([5.0]))
        self.assertIs(m.has_rsample, True)

    def test_has_rsample_with_multiple_params(self):
        """df/loc/scale instances keep has_rsample == True."""
        m = StudentT(
            torch.tensor([5.0]), loc=torch.tensor([1.0]),
            scale=torch.tensor([2.0]))
        self.assertIs(m.has_rsample, True)
        self.assertEqual(tuple(m.batch_shape), (1,))

    def test_rsample_available(self):
        """has_rsample=True implies a working rsample with fixed seed."""
        torch.manual_seed(0)
        m = StudentT(torch.tensor([5.0]))
        s = m.rsample((6,))
        self.assertEqual(tuple(s.shape), (6, 1))
        self.assertTrue(bool(torch.isfinite(s).all()))

    def test_rsample_tracks_grad(self):
        """rsample output tracks grad through the df parameter."""
        df = torch.tensor([5.0], requires_grad=True)
        m = StudentT(df)
        s = m.rsample()
        self.assertTrue(s.requires_grad)
        self.assertTrue(bool(torch.isfinite(s).all()))

    def test_sample_does_not_track_grad(self):
        """sample() blocks grad, unlike the reparameterized rsample."""
        torch.manual_seed(1)
        m = StudentT(torch.tensor([5.0]))
        s = m.sample()
        self.assertFalse(s.requires_grad)


if __name__ == "__main__":
    run_tests()
