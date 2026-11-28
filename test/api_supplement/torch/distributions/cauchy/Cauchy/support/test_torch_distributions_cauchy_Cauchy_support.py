# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.cauchy.Cauchy.support 接口功能正确性
API 名称：torch.distributions.cauchy.Cauchy.support
API 签名：support = constraints.real（类属性，Constraint 对象；源码 cauchy.py:35）

覆盖维度表：
| 覆盖维度     | 说明                                            | 覆盖情况                              |
|--------------|-------------------------------------------------|---------------------------------------|
| 空/非空      | support 为非 None 的 Constraint 单例            | 已覆盖（is_real_constraint）          |
| 枚举选项     | 本接口无枚举入参（连续全实数支持集）            | 不适用                                |
| 参数类型     | 无入参；check 入参为 Tensor                     | 已覆盖（check_* 各用例）              |
| 传参与不传参 | 属性读取形态；check 必传 value                  | 已覆盖（is_real_constraint、check_*） |
| 等价类/边界  | 0/负值/大值/±inf/NaN 各等价类                   | 已覆盖（check_elementwise_pattern）   |
| 正常传参     | 重参数化采样值全部落在实数支持集内              | 已覆盖（check_accepts_rsample）       |
| 异常传参     | NaN 返回 False（不抛异常）                      | 已覆盖（check_elementwise_pattern）   |
| 混合设备类型 | 支持集为纯 Python 约束对象，无设备迁移语义      | 不适用                                |

未覆盖项及原因：
- 无。

注意：本 API 为分布类属性与约束对象契约，CPU 上即可真实执行；断言落在
     「支持集对象同一性 / 连续实数 check 的逐元素真假 / 采样值全满足」
     三类可证伪契约上，不依赖加速器。
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
class TestCauchySupport(TestCase):
    """Contract tests for torch.distributions.cauchy.Cauchy.support."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_support_is_real_constraint(self):
        """support must be the constraints.real singleton for the distribution."""
        dist = torch.distributions.Cauchy(
            torch.tensor([0.0]), torch.tensor([1.0])
        )
        self.assertIs(dist.support, torch.distributions.constraints.real)
        # Upstream contract: real is continuous (not discrete), event_dim 0.
        self.assertIs(dist.support.is_discrete, False)
        self.assertEqual(dist.support.event_dim, 0)

    def test_support_check_accepts_rsample(self):
        """Seeded rsample values must all satisfy the real-line support."""
        torch.manual_seed(13)
        dist = torch.distributions.Cauchy(
            torch.tensor([0.0, 5.0]), torch.tensor([1.0, 2.0])
        )
        samples = dist.rsample((1000,))
        self.assertEqual(samples.shape, (1000, 2))
        self.assertEqual(dist.support.check(samples).all().item(), True)

    def test_support_check_elementwise_pattern(self):
        """check() is elementwise: finite values valid, NaN invalid, inf allowed.

        Upstream _Real.check is `value == value` (constraints.py:424-425), so the
        extended real line admits +/-inf but rejects NaN.
        """
        support = torch.distributions.constraints.real
        values = torch.tensor(
            [0.0, -5.0, 1e10, float('inf'), float('-inf'), float('nan')]
        )
        result = support.check(values)
        expected = torch.tensor([True, True, True, True, True, False])
        self.assertEqual(result.dtype, torch.bool)
        self.assertEqual(result.tolist(), expected.tolist())

    def test_support_check_batch_shape_semantics(self):
        """check() preserves the queried value's batch shape."""
        dist = torch.distributions.Cauchy(
            torch.tensor([0.0]), torch.tensor([1.0])
        )
        values = torch.tensor([[1.0, float('nan')], [0.0, 2.0]])
        result = dist.support.check(values)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.tolist(), [[True, False], [True, True]])


if __name__ == "__main__":
    run_tests()
