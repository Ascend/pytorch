# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.distribution.Distribution.sample 接口功能正确性
API 名称：torch.distributions.distribution.Distribution.sample
API 签名：sample(self, sample_shape: _size = torch.Size()) -> Tensor
         （源码 distribution.py:177-183，内部在 no_grad 下调用 rsample）

覆盖维度表：
| 覆盖维度     | 说明                                              | 覆盖情况                                  |
|--------------|---------------------------------------------------|-------------------------------------------|
| 空/非空      | sample_shape 为空 () 与非空元组两种形态           | 已覆盖（default_shape、batched_shape）    |
| 枚举选项     | sample_shape 接受 tuple / torch.Size 两种容器     | 已覆盖（batched_shape）                   |
| 参数类型     | sample_shape 为 int 元组序列，非 Tensor           | 已覆盖（batched_shape）                   |
| 传参与不传参 | 省略默认 torch.Size() 与显式传入两形态            | 已覆盖（default_shape_equals_explicit）   |
| 等价类/边界  | 空形状、一维、多维 sample_shape 的边界            | 已覆盖（batched_shape）                   |
| 正常传参     | 形状拼接正确、无梯度、同种子可复现                | 已覆盖（detached_from_graph、seeded）     |
| 异常传参     | 抽象基类直接调用抛 NotImplementedError            | 已覆盖（base_distribution_raises）        |
| 混合设备类型 | 分布参数为 CPU 张量，采样无设备迁移语义           | 不适用                                    |

未覆盖项及原因：
- 未覆盖：sample_n(n) 弃用入口。它属于另一个 API（且带 FutureWarning 契约），
  不在本 API 契约面内。

注意：采样类断言只固定种子后核对「形状 + 逐位可复现 + support 内」，
     不把具体采样数值钉死为常量；no_grad 契约用 requires_grad/grad_fn 验证。
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


def _normal_fixture():
    loc = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float64)
    scale = torch.tensor([[1.0, 0.5], [2.0, 1.5]], dtype=torch.float64)
    return torch.distributions.Normal(loc, scale)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestDistributionSample(TestCase):
    """Contract tests for Distribution.sample via the Normal reparameterization."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_default_shape_is_batch_shape(self):
        """sample() with no argument returns one draw per batch entry."""
        dist = _normal_fixture()
        sample = dist.sample()
        self.assertEqual(sample.shape, (2, 2))
        self.assertEqual(dist.sample(torch.Size()).shape, (2, 2))

    def test_batched_sample_shape(self):
        """sample_shape is prepended to the batch shape (extended shape rule)."""
        dist = _normal_fixture()
        self.assertEqual(dist.sample((5,)).shape, (5, 2, 2))
        self.assertEqual(dist.sample((3, 4)).shape, (3, 4, 2, 2))
        self.assertEqual(dist.sample((2, 3, 4)).shape, (2, 3, 4, 2, 2))

    def test_default_shape_equals_explicit_empty(self):
        """sample() and sample(torch.Size()) must consume RNG identically."""
        dist = _normal_fixture()
        torch.manual_seed(42)
        a = dist.sample()
        torch.manual_seed(42)
        b = dist.sample(torch.Size())
        torch.testing.assert_close(a, b)

    def test_sample_detached_from_graph(self):
        """sample() runs under no_grad even for requires_grad parameters."""
        loc = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        scale = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        dist = torch.distributions.Normal(loc, scale)
        sample = dist.sample((4,))
        self.assertIs(sample.requires_grad, False)
        self.assertIsNone(sample.grad_fn)
        # Contrast: the reparameterized path really does carry gradients.
        self.assertIs(dist.rsample((4,)).requires_grad, True)

    def test_sample_reproducible_with_seed(self):
        """Same manual seed must reproduce the sample bit-for-bit."""
        dist = _normal_fixture()
        torch.manual_seed(123)
        first = dist.sample((10,))
        torch.manual_seed(123)
        second = dist.sample((10,))
        torch.testing.assert_close(first, second)
        # Cross-seed draws must differ (the sampler really consumes the RNG).
        torch.manual_seed(124)
        third = dist.sample((10,))
        self.assertEqual(torch.equal(first, third), False)

    def test_samples_inside_support(self):
        """Normal samples must satisfy the real-line support constraint."""
        dist = _normal_fixture()
        torch.manual_seed(5)
        samples = dist.sample((200,))
        self.assertEqual(dist.support.check(samples).all().item(), True)

    def test_base_distribution_raises_not_implemented(self):
        """The abstract base class has no rsample, so sample() must raise."""
        base = torch.distributions.Distribution(validate_args=False)
        with self.assertRaises(NotImplementedError):
            base.sample()


if __name__ == "__main__":
    run_tests()
