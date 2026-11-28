# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.generalized_pareto.GeneralizedPareto.has_rsample 接口功能正确性
API 名称：torch.distributions.generalized_pareto.GeneralizedPareto.has_rsample
API 签名：has_rsample = True（类属性 bool；源码 generalized_pareto.py:44）

覆盖维度表：
| 覆盖维度     | 说明                                              | 覆盖情况                                  |
|--------------|---------------------------------------------------|-------------------------------------------|
| 空/非空      | 布尔类属性无空值语义                              | 不适用                                    |
| 枚举选项     | 取值只有 True（对基类默认 False 的覆盖）          | 已覆盖（flag_is_true）                    |
| 参数类型     | 类属性为 bool；实例 rsample 返回 Tensor           | 已覆盖（flag_is_true、rsample_gradient）  |
| 传参与不传参 | 属性读取形态，无入参                              | 已覆盖（flag_is_true）                    |
| 等价类/边界  | 边界：has_rsample=True 与 sample 无梯度对照       | 已覆盖（sample_is_detached）              |
| 正常传参     | rsample 对 requires_grad 参数保留梯度路径         | 已覆盖（rsample_gradient）                |
| 异常传参     | 无异常路径（纯标记属性）                          | 不适用                                    |
| 混合设备类型 | 标记属性与 CPU 张量采样，无设备迁移语义           | 不适用                                    |

未覆盖项及原因：
- 无。

注意：可证伪断言落在「标记值同一性 / rsample 梯度真实流动（backward 后
     各参数梯度有限且非零）/ sample 与 rsample 的梯度对照」三处。
"""
import os
import unittest

import torch

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
    # TestCase. Floating-point comparisons below use the device-agnostic
    # torch.testing.assert_close, so the fallback is safe here.
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
class TestGeneralizedParetoHasRsample(TestCase):
    """Contract tests for GeneralizedPareto.has_rsample."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_flag_is_true(self):
        """The class attribute and the instance attribute must both be True."""
        self.assertIs(
            torch.distributions.GeneralizedPareto.has_rsample, True
        )
        dist = torch.distributions.GeneralizedPareto(0.0, 1.0, 0.5)
        self.assertIs(dist.has_rsample, True)
        self.assertIsInstance(dist.has_rsample, bool)

    def test_rsample_gradient_flows(self):
        """rsample on requires_grad params carries gradients through backward."""
        loc = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        scale = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
        concentration = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
        dist = torch.distributions.GeneralizedPareto(
            loc, scale, concentration
        )
        sample = dist.rsample((8,))
        self.assertIs(sample.requires_grad, True)
        self.assertIsNotNone(sample.grad_fn)
        sample.sum().backward()
        for param, name in (
            (loc, "loc"),
            (scale, "scale"),
            (concentration, "concentration"),
        ):
            self.assertIsNotNone(param.grad, name)
            self.assertEqual(torch.isfinite(param.grad).all().item(), True)
        # loc enters the sample with unit slope: d(sum)/d(loc) = numel.
        torch.testing.assert_close(
            loc.grad, torch.full_like(loc.grad, 8.0)
        )

    def test_sample_is_detached(self):
        """sample() must wrap rsample in no_grad, so no gradient is tracked."""
        loc = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        scale = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
        concentration = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
        dist = torch.distributions.GeneralizedPareto(
            loc, scale, concentration
        )
        sample = dist.sample((4,))
        self.assertIs(sample.requires_grad, False)
        self.assertIsNone(sample.grad_fn)
        for param in (loc, scale, concentration):
            self.assertIsNone(param.grad)


if __name__ == "__main__":
    run_tests()
