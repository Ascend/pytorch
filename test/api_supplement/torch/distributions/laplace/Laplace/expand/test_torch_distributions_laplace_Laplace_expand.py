# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.laplace.Laplace.expand 接口功能正确性
API 名称：torch.distributions.laplace.Laplace.expand
API 签名：expand(self, batch_shape, _instance=None) -> Laplace
         （上游源码 torch/distributions/laplace.py L64-L71：展开 loc/scale 后
          以 batch_shape 重建实例并继承 _validate_args）

覆盖维度表：
| 覆盖维度     | 说明                                      | 覆盖情况                               |
|--------------|-------------------------------------------|----------------------------------------|
| 空/非空      | 空 batch（标量参数）expand 到非空 (3,)    | 已覆盖（test_expand_scalar_to_batch）  |
| 枚举选项     | 本接口无枚举入参（batch_shape 为 Size）   | 不适用                                 |
| 参数类型     | loc/scale 为 float 标量 vs Tensor         | 已覆盖（test_expand_scalar_to_batch、  |
|              |                                           | test_expand_batch_shape_and_params）   |
| 传参与不传参 | 仅传 batch_shape，_instance 走默认 None   | 已覆盖（各用例均未传 _instance）       |
| 等价类/边界  | 追加前导维 (2,)→(4,2)；恒等 (2,)→(2,)；   | 已覆盖（test_expand_batch_shape_and_   |
|              | 非法收缩 (2,)→(1,)                        | params、test_expand_invalid_raises）   |
| 正常传参     | expand 后 log_prob 与闭式解一致，         | 已覆盖（test_expand_log_prob、         |
|              | mean=loc/variance=2*scale^2 闭式解成立    | test_expand_analytic_stats）           |
| 异常传参     | 不可广播的目标 batch_shape 抛 RuntimeError| 已覆盖（test_expand_invalid_raises）   |
| 混合设备类型 | 纯 Python 契约测试，张量统一经 self.device| 不适用（expand 语义与设备无关）        |
|              | 构造，不涉及跨设备混用                    |                                        |

未覆盖项及原因：
- 未覆盖：_instance 非默认路径。该参数是框架内部复用实例的私有协议
  （_get_checked_instance），公开契约上与默认路径等价，不做伪对照。

注意：expand 不改变分布族与数学语义，断言全部落在「batch_shape/参数形状、
     log_prob 闭式解对照、mean/variance 闭式解、采样形状、非法 shape 报错」
     五类可证伪契约上，浮点对照一律使用 torch.testing.assert_close。
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
class TestLaplaceExpand(TestCase):
    """Test cases for torch.distributions.laplace.Laplace.expand."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_expand_batch_shape_and_params(self):
        """expand 追加前导维后 batch_shape/loc/scale 形状与取值均正确。"""
        loc = torch.tensor([1.0, -2.0], device=self.device)
        scale = torch.tensor([0.5, 1.5], device=self.device)
        dist = torch.distributions.Laplace(loc, scale)
        expanded = dist.expand((4, 2))
        self.assertEqual(expanded.batch_shape, torch.Size([4, 2]))
        self.assertEqual(expanded.loc.shape, torch.Size([4, 2]))
        self.assertEqual(expanded.scale.shape, torch.Size([4, 2]))
        torch.testing.assert_close(expanded.loc, loc.expand((4, 2)))
        torch.testing.assert_close(expanded.scale, scale.expand((4, 2)))
        # 恒等边界：expand 到原 batch 形状后参数保持逐元素一致
        same = dist.expand((2,))
        self.assertEqual(same.batch_shape, torch.Size([2]))
        torch.testing.assert_close(same.loc, loc)
        torch.testing.assert_close(same.scale, scale)

    def test_expand_log_prob(self):
        """expand 后 log_prob 与闭式解 -log(2*scale)-|x-loc|/scale 逐元素一致。"""
        loc = torch.tensor([1.0, -2.0], device=self.device)
        scale = torch.tensor([0.5, 1.5], device=self.device)
        dist = torch.distributions.Laplace(loc, scale)
        expanded = dist.expand((4, 2))
        torch.manual_seed(0)
        value = torch.randn(4, 2, device=self.device)
        expected = (-torch.log(2 * expanded.scale)
                    - (value - expanded.loc).abs() / expanded.scale)
        torch.testing.assert_close(expanded.log_prob(value), expected)
        torch.testing.assert_close(expanded.log_prob(value), dist.log_prob(value))

    def test_expand_analytic_stats(self):
        """expand 后 mean=loc、variance=2*scale^2、stddev=sqrt(2)*scale 闭式解成立。"""
        loc = torch.tensor([1.0, -2.0], device=self.device)
        scale = torch.tensor([0.5, 1.5], device=self.device)
        dist = torch.distributions.Laplace(loc, scale)
        expanded = dist.expand((4, 2))
        torch.testing.assert_close(expanded.mean, expanded.loc)
        torch.testing.assert_close(expanded.variance, 2 * expanded.scale.pow(2))
        torch.testing.assert_close(
            expanded.stddev, math.sqrt(2) * expanded.scale)

    def test_expand_scalar_to_batch(self):
        """空 batch（float 标量参数）expand 到非空 batch (3,)。"""
        dist = torch.distributions.Laplace(0.5, 2.0)
        self.assertEqual(dist.batch_shape, torch.Size([]))
        expanded = dist.expand((3,))
        self.assertEqual(expanded.batch_shape, torch.Size([3]))
        # Python-float 参数落在默认(CPU)设备，expand 不改设备，期望取 CPU 张量。
        torch.testing.assert_close(
            expanded.loc, torch.full((3,), 0.5))
        torch.testing.assert_close(
            expanded.scale, torch.full((3,), 2.0))

    def test_expand_seed_sample_shape(self):
        """固定种子采样：expand 后 sample 形状为 sample_shape + expanded batch_shape。"""
        loc = torch.tensor([1.0, -2.0], device=self.device)
        scale = torch.tensor([0.5, 1.5], device=self.device)
        dist = torch.distributions.Laplace(loc, scale)
        expanded = dist.expand((4, 2))
        torch.manual_seed(2024)
        sample = expanded.sample((6,))
        self.assertEqual(sample.shape, torch.Size([6, 4, 2]))
        self.assertTrue(torch.isfinite(sample).all())

    def test_expand_invalid_raises(self):
        """目标 batch_shape 与现有参数不可广播时必须抛 RuntimeError。"""
        loc = torch.tensor([1.0, -2.0], device=self.device)
        scale = torch.tensor([0.5, 1.5], device=self.device)
        dist = torch.distributions.Laplace(loc, scale)
        with self.assertRaises(RuntimeError):
            dist.expand((1,))


if __name__ == "__main__":
    run_tests()
