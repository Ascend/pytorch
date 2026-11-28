# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.bernoulli.Bernoulli.support 接口功能正确性
API 名称：torch.distributions.bernoulli.Bernoulli.support
API 签名：support = constraints.boolean（类属性，Constraint 对象；源码 bernoulli.py:43）

覆盖维度表：
| 覆盖维度     | 说明                                          | 覆盖情况                                  |
|--------------|-----------------------------------------------|-------------------------------------------|
| 空/非空      | support 为非 None 的 Constraint 单例          | 已覆盖（is_boolean_constraint）           |
| 枚举选项     | 支持集为离散枚举 {0, 1}                       | 已覆盖（check_accepts_valid_samples 等）  |
| 参数类型     | 无入参；check 入参为 Tensor                   | 已覆盖（check_* 各用例）                  |
| 传参与不传参 | 属性读取形态；check 必传 value                | 已覆盖（is_boolean_constraint、check_*）  |
| 等价类/边界  | 边界值 0/1、越界 2/0.5/-1、NaN                | 已覆盖（check_elementwise_pattern 等）    |
| 正常传参     | 采样值全部满足支持集约束                      | 已覆盖（check_accepts_valid_samples）     |
| 异常传参     | 越界值 check 返回 False（不抛异常）           | 已覆盖（check_rejects_out_of_support）    |
| 混合设备类型 | 支持集为纯 Python 约束对象，无设备迁移语义    | 不适用                                    |

未覆盖项及原因：
- 无。

注意：本 API 为分布类属性与约束对象契约，CPU 上用固定种子张量即可真实执行；
     断言落在「支持集对象同一性 / 离散枚举 check 的逐元素真假 / 大样本全满足」
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
class TestBernoulliSupport(TestCase):
    """Contract tests for torch.distributions.bernoulli.Bernoulli.support."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_support_is_boolean_constraint(self):
        """support must be the constraints.boolean singleton for both param forms."""
        dist_probs = torch.distributions.Bernoulli(probs=torch.tensor([0.3, 0.9]))
        dist_logits = torch.distributions.Bernoulli(logits=torch.tensor([0.1, -1.0]))
        self.assertIs(dist_probs.support, torch.distributions.constraints.boolean)
        self.assertIs(dist_logits.support, torch.distributions.constraints.boolean)
        # Upstream contract: boolean is discrete with event_dim 0.
        self.assertIs(dist_probs.support.is_discrete, True)
        self.assertEqual(dist_probs.support.event_dim, 0)

    def test_support_check_accepts_valid_samples(self):
        """Large seeded sample set must lie entirely inside the {0, 1} support."""
        torch.manual_seed(7)
        dist = torch.distributions.Bernoulli(probs=torch.tensor([0.3, 0.9]))
        samples = dist.sample((10000,))
        self.assertEqual(samples.shape, (10000, 2))
        valid = dist.support.check(samples)
        self.assertEqual(valid.dtype, torch.bool)
        self.assertEqual(valid.all().item(), True)
        # The samples themselves must be an exact subset of {0, 1}.
        unique = torch.unique(samples)
        self.assertEqual(unique.numel(), 2)
        self.assertEqual(set(unique.tolist()), {0.0, 1.0})
        # Both outcomes really occur for the chosen probs (0.3 / 0.9).
        self.assertEqual((samples == 1).any().item(), True)
        self.assertEqual((samples == 0).any().item(), True)

    def test_support_check_elementwise_pattern(self):
        """check() must be elementwise: 0/1 valid, 2 / 0.5 / -1 / NaN invalid."""
        support = torch.distributions.constraints.boolean
        values = torch.tensor([0.0, 1.0, 2.0, 0.5, -1.0, float('nan')])
        result = support.check(values)
        expected = torch.tensor([True, True, False, False, False, False])
        self.assertEqual(result.dtype, torch.bool)
        self.assertEqual(result.tolist(), expected.tolist())

    def test_support_check_batch_shape_semantics(self):
        """check() preserves batch shape of the queried value."""
        dist = torch.distributions.Bernoulli(probs=torch.tensor([0.25, 0.75]))
        values = torch.tensor([[0.0, 1.0], [3.0, 0.0], [1.0, 0.5]])
        result = dist.support.check(values)
        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(result.tolist(), [[True, True], [False, True], [True, False]])


if __name__ == "__main__":
    run_tests()
