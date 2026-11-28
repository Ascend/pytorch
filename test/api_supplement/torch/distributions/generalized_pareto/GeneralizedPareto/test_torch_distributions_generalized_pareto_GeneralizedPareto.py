# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.generalized_pareto.GeneralizedPareto 接口功能正确性
API 名称：torch.distributions.generalized_pareto.GeneralizedPareto
API 签名：GeneralizedPareto(loc, scale, concentration, validate_args=None)
         （源码 generalized_pareto.py:46-58）

覆盖维度表：
| 覆盖维度     | 说明                                              | 覆盖情况                                  |
|--------------|---------------------------------------------------|-------------------------------------------|
| 空/非空      | 参数不可省略（三参必传）；validate_args 可缺省    | 已覆盖（broadcast_and_batch_shape 等）    |
| 枚举选项     | validate_args 的 True/False/None 三种取值         | 已覆盖（validate_args_* 两用例）          |
| 参数类型     | 全标量数、张量、标量与张量混合三种入参类型        | 已覆盖（broadcast_and_batch_shape）       |
| 传参与不传参 | validate_args 传与不传（默认 None）两形态         | 已覆盖（broadcast、validate_args 用例）   |
| 等价类/边界  | 标量 () batch、广播 batch、conc=0 指数退化        | 已覆盖（scalar_batch、exponential_case）  |
| 正常传参     | 形状/参数广播正确、固定种子采样落在支持集内       | 已覆盖（sample_within_support、seeded）   |
| 异常传参     | scale<=0 触发构造期 ValueError                    | 已覆盖（validate_args_rejects_bad_scale） |
| 混合设备类型 | 参数为 CPU 张量，构造无设备迁移语义               | 不适用                                    |

未覆盖项及原因：
- 无。log_prob/cdf/expand 等方法契约由本目录各自独立文件覆盖。

注意：采样只断言「固定种子下形状正确、可复现、落在支持集内」，不把采样
     数值钉死为常量。
"""
import math
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
class TestGeneralizedPareto(TestCase):
    """Contract tests for the GeneralizedPareto constructor."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_broadcast_and_batch_shape(self):
        """Parameters are broadcast to a common shape; batch_shape follows loc."""
        dist = torch.distributions.GeneralizedPareto(
            torch.tensor([0.0, 1.0], dtype=torch.float64),
            torch.tensor(2.0, dtype=torch.float64),
            torch.tensor([0.5, -0.3], dtype=torch.float64),
        )
        self.assertEqual(dist.batch_shape, torch.Size((2,)))
        self.assertEqual(dist.loc.shape, (2,))
        self.assertEqual(dist.scale.shape, (2,))
        self.assertEqual(dist.concentration.shape, (2,))
        torch.testing.assert_close(dist.scale, torch.full((2,), 2.0, dtype=torch.float64))

    def test_scalar_batch_shape(self):
        """Three plain scalars produce a scalar batch shape ()."""
        dist = torch.distributions.GeneralizedPareto(0.5, 2.0, 0.4)
        self.assertEqual(dist.batch_shape, torch.Size())
        self.assertEqual(dist.event_shape, torch.Size())

    def test_validate_args_rejects_bad_scale(self):
        """scale must be positive: 0 or negative values raise at construction."""
        with self.assertRaises(ValueError):
            torch.distributions.GeneralizedPareto(0.0, -1.0, 0.5, validate_args=True)
        with self.assertRaises(ValueError):
            torch.distributions.GeneralizedPareto(
                torch.tensor([0.0]),
                torch.tensor([0.0]),
                torch.tensor([0.5]),
                validate_args=True,
            )
        # Opting out must be honored on the same invalid inputs.
        torch.distributions.GeneralizedPareto(0.0, -1.0, 0.5, validate_args=False)

    def test_sample_within_support(self):
        """Seeded samples lie inside the per-instance support (lower bound loc)."""
        torch.manual_seed(29)
        dist = torch.distributions.GeneralizedPareto(
            torch.tensor(0.5, dtype=torch.float64),
            torch.tensor(1.5, dtype=torch.float64),
            torch.tensor(0.4, dtype=torch.float64),
        )
        samples = dist.sample((2000,))
        self.assertEqual(samples.shape, (2000,))
        self.assertEqual((samples >= dist.loc).all().item(), True)
        self.assertEqual(dist.support.check(samples).all().item(), True)

    def test_sample_reproducible_with_seed(self):
        """Same manual seed must reproduce the sample bit-for-bit."""
        dist = torch.distributions.GeneralizedPareto(0.0, 1.0, 0.3)
        torch.manual_seed(31)
        first = dist.sample((10,))
        torch.manual_seed(31)
        second = dist.sample((10,))
        torch.testing.assert_close(first, second)
        torch.manual_seed(32)
        third = dist.sample((10,))
        self.assertEqual(torch.equal(first, third), False)

    def test_exponential_special_case(self):
        """loc=0, concentration=0 degenerates to Exponential(rate=1/scale)."""
        scale = 2.5
        dist = torch.distributions.GeneralizedPareto(0.0, scale, 0.0)
        # float32 与 dist 参数 dtype 一致（float64 x 会触发混合精度计算，
        # 与 float32 内部链路产生 ~3e-8 的表示差）。
        x = torch.linspace(0.1, 5.0, 9)
        expected = -x / scale - math.log(scale)
        # float32 exp/log 链的表示误差（~3e-8）远大于 float64 的 1e-12；
        # 语义错误（非指数退化）差异为 O(1)，容差放宽不损可证伪性。
        torch.testing.assert_close(
            dist.log_prob(x), expected, atol=1e-6, rtol=1e-6
        )


if __name__ == "__main__":
    run_tests()
