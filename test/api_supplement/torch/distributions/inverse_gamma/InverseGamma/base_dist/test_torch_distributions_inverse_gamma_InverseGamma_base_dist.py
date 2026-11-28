# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.inverse_gamma.InverseGamma.base_dist 接口功能正确性
API 名称：torch.distributions.inverse_gamma.InverseGamma.base_dist
API 签名：base_dist 实例属性，类型为 Gamma
         （上游源码 torch/distributions/inverse_gamma.py L52-L56：base_dist =
          Gamma(concentration, rate)，再经 PowerTransform(-1) 变换 Y=1/X）

覆盖维度表：
| 覆盖维度     | 说明                                      | 覆盖情况                               |
|--------------|-------------------------------------------|----------------------------------------|
| 空/非空      | 标量参数空 batch；batched 参数非空 batch  | 已覆盖（test_base_dist_type_and_params、|
|              | base_dist.batch_shape 均正确              | test_base_dist_batch_shape）           |
| 枚举选项     | 本接口无枚举入参                          | 不适用                                 |
| 参数类型     | concentration/rate 传 Tensor（含 batched）| 已覆盖（test_base_dist_type_and_params、|
|              | 与 float 标量                             | test_base_dist_batch_shape、           |
|              |                                           | test_base_dist_float_scalar_params）   |
| 传参与不传参 | 无参属性，仅默认读取                      | 已覆盖（各用例均直接读属性）           |
| 等价类/边界  | concentration=1 边界：mean 公式经         | 已覆盖（test_base_dist_params_feed_    |
|              | torch.where 落到 inf 分支                 | analytic_mean）                        |
| 正常传参     | base_dist 类型为 Gamma，concentration/rate| 已覆盖（test_base_dist_type_and_params、|
|              | 与构造参数一致，log_prob 与 Gamma 闭式解  | test_base_dist_log_prob_closed_form、  |
|              | 及 Y=1/X 变换关系一致                     | test_base_dist_log_prob_relation）     |
| 异常传参     | 无入参属性，不存在异常传参路径            | 不适用                                 |
| 混合设备类型 | 纯 Python 契约测试，张量统一经 self.device| 不适用（属性语义与设备无关）           |
|              | 构造，不涉及跨设备混用                    |                                        |

未覆盖项及原因：
- 未覆盖：validate_args=True 时构造非法参数的行为。该路径属于构造器
  （__init__）的参数校验契约而非 base_dist 属性契约，不在本文件范围内伪对照。

注意：base_dist 是 InverseGamma 的幂变换（PowerTransform(-1)）基分布，断言
     全部落在「返回类型 isinstance(Gamma)、参数一致性、Gamma 闭式解、
     变换雅可比关系 log p_Y(y)=log p_X(1/y)-2*log(y)、mean 边界分支」五类
     可证伪契约上，浮点对照一律使用 torch.testing.assert_close。
"""
import math
import os
import unittest

import torch
import torch.distributions  # noqa: F401  (被测 API 所在包)

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


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestInverseGammaBaseDist(TestCase):
    """Test cases for torch.distributions.inverse_gamma.InverseGamma.base_dist."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_base_dist_type_and_params(self):
        """base_dist 是 Gamma 实例：concentration/rate 与构造参数一致（标量张量构造）。"""
        concentration = torch.tensor(2.0, device=self.device)
        rate = torch.tensor(3.0, device=self.device)
        dist = torch.distributions.InverseGamma(concentration, rate)
        base = dist.base_dist
        self.assertIsInstance(base, torch.distributions.Gamma)
        torch.testing.assert_close(base.concentration, concentration)
        torch.testing.assert_close(base.rate, rate)
        # InverseGamma.concentration/rate 透传 base_dist（源码 L62-L68）
        torch.testing.assert_close(dist.concentration, base.concentration)
        torch.testing.assert_close(dist.rate, base.rate)

    def test_base_dist_float_scalar_params(self):
        """float 标量构造：base_dist 参数经 torch.as_tensor 落在默认(CPU)设备并透传。"""
        dist = torch.distributions.InverseGamma(2.0, 3.0)
        base = dist.base_dist
        self.assertIsInstance(base, torch.distributions.Gamma)
        self.assertEqual(dist.batch_shape, torch.Size([]))
        # Python-float 参数落在默认(CPU)设备，期望值同取 CPU 张量。
        torch.testing.assert_close(base.concentration, torch.full((), 2.0))
        torch.testing.assert_close(base.rate, torch.full((), 3.0))
        # 可证伪锚点：Gamma(2,3).log_prob(0.5) 的解析值 log(0.5)-1.5+2*log(3)。
        expected = math.log(0.5) - 3.0 * 0.5 + 2.0 * math.log(3.0)
        torch.testing.assert_close(
            base.log_prob(torch.tensor(0.5)), torch.full((), expected))

    def test_base_dist_log_prob_closed_form(self):
        """base_dist.log_prob 与 Gamma(rate 参数化) 闭式解
        (a-1)*log(x)-b*x+a*log(b)-lgamma(a) 逐元素一致。"""
        concentration = torch.tensor([2.0, 3.0], device=self.device)
        rate = torch.tensor([3.0, 1.5], device=self.device)
        dist = torch.distributions.InverseGamma(concentration, rate)
        base = dist.base_dist
        x = torch.tensor([[0.5, 1.0], [2.0, 0.25]], device=self.device)
        # Gamma(rate) 密度 b^a/gamma(a) * x^(a-1) * e^(-b*x)：rate 乘在 x 上、
        # a*log(b) 为正号——对应上游 Gamma.log_prob 的实现公式。
        expected = ((concentration - 1) * x.log() - rate * x
                    + concentration * rate.log() - concentration.lgamma())
        torch.testing.assert_close(base.log_prob(x), expected)

    def test_base_dist_log_prob_relation(self):
        """变换关系：log p_Y(y) = log p_X(1/y) - 2*log(y)（Y=1/X 的雅可比）。"""
        concentration = torch.tensor([2.0, 3.0], device=self.device)
        rate = torch.tensor([3.0, 1.5], device=self.device)
        dist = torch.distributions.InverseGamma(concentration, rate)
        base = dist.base_dist
        y = torch.tensor([[0.5, 1.0], [2.0, 4.0]], device=self.device)
        expected = base.log_prob(1.0 / y) - 2 * y.log()
        torch.testing.assert_close(dist.log_prob(y), expected)

    def test_base_dist_batch_shape(self):
        """batched 参数：base_dist 的 batch_shape/concentration/rate 形状与构造参数一致。"""
        concentration = torch.tensor([2.0, 3.0], device=self.device)
        rate = torch.tensor([3.0, 1.5], device=self.device)
        dist = torch.distributions.InverseGamma(concentration, rate)
        base = dist.base_dist
        self.assertEqual(base.batch_shape, torch.Size([2]))
        torch.testing.assert_close(base.concentration, concentration)
        torch.testing.assert_close(base.rate, rate)

    def test_base_dist_params_feed_analytic_mean(self):
        """mean=rate/(concentration-1)；concentration=1 边界按源码落到 inf 分支。"""
        concentration = torch.tensor([2.0, 1.0], device=self.device)
        rate = torch.tensor([3.0, 1.0], device=self.device)
        dist = torch.distributions.InverseGamma(concentration, rate)
        mean = dist.mean
        torch.testing.assert_close(mean[0], torch.tensor(3.0, device=self.device))
        self.assertTrue(torch.isinf(mean[1]))


if __name__ == "__main__":
    run_tests()
