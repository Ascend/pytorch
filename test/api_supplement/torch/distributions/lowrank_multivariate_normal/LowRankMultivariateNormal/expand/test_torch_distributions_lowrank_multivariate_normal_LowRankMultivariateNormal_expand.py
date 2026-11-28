# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.lowrank_multivariate_normal.
         LowRankMultivariateNormal.expand 接口功能正确性
API 名称：torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.expand
API 签名：expand(self, batch_shape, _instance=None) -> LowRankMultivariateNormal
         （上游源码 torch/distributions/lowrank_multivariate_normal.py
          L141-L155：展开 loc/cov_diag/cov_factor 并继承 unbroadcasted 参数）

覆盖维度表：
| 覆盖维度     | 说明                                      | 覆盖情况                               |
|--------------|-------------------------------------------|----------------------------------------|
| 空/非空      | 空 batch（batch_shape=()）expand 到非空   | 已覆盖（test_expand_empty_to_batch）   |
|              | (4,)                                      |                                        |
| 枚举选项     | 本接口无枚举入参（batch_shape 为 Size）   | 不适用                                 |
| 参数类型     | loc/cov_factor/cov_diag 均为 Tensor       | 已覆盖（test_expand_batch_shape_and_   |
|              |                                           | params、test_expand_consistency）      |
| 传参与不传参 | 仅传 batch_shape，_instance 走默认 None   | 已覆盖（各用例均未传 _instance）       |
| 等价类/边界  | 追加前导维 ()→(4,)；恒等 (4,)→(4,)；非法  | 已覆盖（test_expand_batch_shape_and_   |
|              | 收缩 (4,)→(5,)                            | params、test_expand_invalid_raises）   |
| 正常传参     | expand 后 covariance_matrix/log_prob 与   | 已覆盖（test_expand_consistency）      |
|              | 原分布逐元素一致                          |                                        |
| 异常传参     | 不可广播的目标 batch_shape 抛 RuntimeError| 已覆盖（test_expand_invalid_raises）   |
| 混合设备类型 | 纯 Python 契约测试，张量统一经 self.device| 不适用（expand 语义与设备无关）        |
|              | 构造，不涉及跨设备混用                    |                                        |

未覆盖项及原因：
- 未覆盖：_instance 非默认路径。该参数是框架内部复用实例的私有协议
  （_get_checked_instance），公开契约上与默认路径等价，不做伪对照。

注意：expand 不改变 event 维与协方差结构，断言全部落在「batch_shape 与
     参数形状、covariance_matrix/log_prob 一致性、采样形状、非法 shape 报错」
     四类可证伪契约上，浮点对照一律使用 torch.testing.assert_close。
"""
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
class TestLowRankMultivariateNormalExpand(TestCase):
    """Test cases for LowRankMultivariateNormal.expand."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_dist(self):
        loc = torch.zeros(2, device=self.device)
        cov_factor = torch.tensor([[1.0], [0.0]], device=self.device)
        cov_diag = torch.ones(2, device=self.device)
        return torch.distributions.LowRankMultivariateNormal(loc, cov_factor, cov_diag)

    def test_expand_batch_shape_and_params(self):
        """expand 追加前导维后 batch_shape/loc/cov_diag/cov_factor 形状与取值正确。"""
        dist = self._make_dist()
        self.assertEqual(dist.batch_shape, torch.Size([]))
        expanded = dist.expand((4,))
        self.assertEqual(expanded.batch_shape, torch.Size([4]))
        self.assertEqual(expanded.event_shape, torch.Size([2]))
        self.assertEqual(expanded.loc.shape, torch.Size([4, 2]))
        self.assertEqual(expanded.cov_diag.shape, torch.Size([4, 2]))
        self.assertEqual(expanded.cov_factor.shape, torch.Size([4, 2, 1]))
        torch.testing.assert_close(
            expanded.loc, dist.loc.expand((4, 2)))
        torch.testing.assert_close(
            expanded.cov_diag, dist.cov_diag.expand((4, 2)))
        # 恒等边界：expand 到原 batch 形状后 loc 保持逐元素一致
        same = expanded.expand((4,))
        torch.testing.assert_close(same.loc, expanded.loc)

    def test_expand_consistency(self):
        """expand 后 covariance_matrix/log_prob 与原分布逐元素一致。"""
        dist = self._make_dist()
        expanded = dist.expand((4,))
        torch.testing.assert_close(
            expanded.covariance_matrix, dist.covariance_matrix.expand((4, 2, 2)))
        torch.manual_seed(0)
        value = torch.randn(4, 2, device=self.device)
        torch.testing.assert_close(expanded.log_prob(value), dist.log_prob(value))
        # 可证伪锚点：log_prob 形状必须为 expanded batch_shape
        self.assertEqual(expanded.log_prob(value).shape, torch.Size([4]))

    def test_expand_empty_to_batch(self):
        """空 batch expand 到非空 batch，event 维保持不变。"""
        dist = self._make_dist()
        expanded = dist.expand((4,))
        self.assertEqual(dist.event_shape, expanded.event_shape)
        self.assertEqual(expanded.batch_shape, torch.Size([4]))

    def test_expand_seed_sample_shape(self):
        """固定种子采样：expand 后 rsample 形状为 sample_shape + expanded batch + event。"""
        dist = self._make_dist()
        expanded = dist.expand((4,))
        torch.manual_seed(2024)
        sample = expanded.rsample((3,))
        self.assertEqual(sample.shape, torch.Size([3, 4, 2]))
        self.assertTrue(torch.isfinite(sample).all())

    def test_expand_invalid_raises(self):
        """目标 batch_shape 与现有参数不可广播时必须抛 RuntimeError。"""
        dist = self._make_dist()
        expanded = dist.expand((4,))
        with self.assertRaises(RuntimeError):
            expanded.expand((5,))


if __name__ == "__main__":
    run_tests()
