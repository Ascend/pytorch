# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.independent.Independent.expand 接口功能正确性
API 名称：torch.distributions.independent.Independent.expand
API 签名：expand(self, batch_shape, _instance=None) -> Independent
         （上游源码 torch/distributions/independent.py L71-L82：展开基分布到
          batch_shape + event_shape[:reinterpreted_batch_ndims]，再重建维度）

覆盖维度表：
| 覆盖维度     | 说明                                      | 覆盖情况                               |
|--------------|-------------------------------------------|----------------------------------------|
| 空/非空      | 空 batch（标量批基分布）expand 到非空 (5,)| 已覆盖（test_expand_empty_to_batch）   |
| 枚举选项     | 本接口无枚举入参（batch_shape 为 Size）   | 不适用                                 |
| 参数类型     | 基分布 loc/scale 为 Tensor（含 batched）  | 已覆盖（test_expand_batch_shape_and_   |
|              |                                           | base_dist、test_expand_log_prob）      |
| 传参与不传参 | 仅传 batch_shape，_instance 走默认 None   | 已覆盖（各用例均未传 _instance）       |
| 等价类/边界  | 追加前导维 (4,)→(6,4)；恒等 (4,)→(4,)；   | 已覆盖（test_expand_batch_shape_and_   |
|              | 非法收缩 (6,4)→(2,)                       | base_dist、test_expand_invalid_raises）|
| 正常传参     | expand 后 log_prob 与原分布逐元素一致，   | 已覆盖（test_expand_log_prob、         |
|              | base_dist.batch_shape 按 ndims 重组       | test_expand_batch_shape_and_base_dist）|
| 异常传参     | 不可广播的目标 batch_shape 抛 RuntimeError| 已覆盖（test_expand_invalid_raises）   |
| 混合设备类型 | 纯 Python 契约测试，张量统一经 self.device| 不适用（expand 语义与设备无关）        |
|              | 构造，不涉及跨设备混用                    |                                        |

未覆盖项及原因：
- 未覆盖：_instance 非默认路径。该参数是框架内部复用实例的私有协议
  （_get_checked_instance），公开契约上与默认路径等价，不做伪对照。

注意：expand 不改变维度重解释语义，断言全部落在「batch_shape 与
     base_dist.batch_shape 重组公式、log_prob 一致性、采样形状、非法 shape
     报错」四类可证伪契约上，浮点对照一律使用 torch.testing.assert_close。
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
class TestIndependentExpand(TestCase):
    """Test cases for torch.distributions.independent.Independent.expand."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_expand_batch_shape_and_base_dist(self):
        """expand 后 batch_shape 正确，base_dist 被展开为 batch_shape+重解释维度。"""
        base = torch.distributions.Normal(
            torch.zeros(4, 3, device=self.device),
            torch.ones(4, 3, device=self.device))
        ind = torch.distributions.Independent(base, 1)
        expanded = ind.expand((6, 4))
        self.assertEqual(expanded.batch_shape, torch.Size([6, 4]))
        self.assertEqual(expanded.event_shape, torch.Size([3]))
        self.assertEqual(expanded.reinterpreted_batch_ndims, 1)
        # 源码公式：base_dist.expand(batch_shape + event_shape[:ndims])
        self.assertEqual(expanded.base_dist.batch_shape, torch.Size([6, 4, 3]))
        # 恒等边界：expand 到原 batch 形状后 base_dist 维度不变
        same = ind.expand((4,))
        self.assertEqual(same.base_dist.batch_shape, torch.Size([4, 3]))
        torch.testing.assert_close(same.base_dist.loc, base.loc)

    def test_expand_log_prob(self):
        """expand 后 log_prob 与原分布逐元素一致（原分布对 value 广播）。"""
        base = torch.distributions.Normal(
            torch.zeros(4, 3, device=self.device),
            torch.ones(4, 3, device=self.device))
        ind = torch.distributions.Independent(base, 1)
        expanded = ind.expand((6, 4))
        torch.manual_seed(0)
        value = torch.randn(6, 4, 3, device=self.device)
        torch.testing.assert_close(expanded.log_prob(value), ind.log_prob(value))
        # 可证伪锚点：与基分布逐元素 log_prob 求和的解析关系一致
        expected = expanded.base_dist.log_prob(value).sum(-1)
        torch.testing.assert_close(expanded.log_prob(value), expected)

    def test_expand_empty_to_batch(self):
        """空 batch（标量批基分布）expand 到非空 batch (5,)。"""
        base = torch.distributions.Normal(
            torch.zeros(3, device=self.device), torch.ones(3, device=self.device))
        ind = torch.distributions.Independent(base, 1)
        self.assertEqual(ind.batch_shape, torch.Size([]))
        expanded = ind.expand((5,))
        self.assertEqual(expanded.batch_shape, torch.Size([5]))
        self.assertEqual(expanded.base_dist.batch_shape, torch.Size([5, 3]))

    def test_expand_seed_sample_shape(self):
        """固定种子采样：expand 后 rsample 形状为 sample_shape + expanded batch_shape + event。"""
        base = torch.distributions.Normal(
            torch.zeros(4, 3, device=self.device),
            torch.ones(4, 3, device=self.device))
        ind = torch.distributions.Independent(base, 1)
        expanded = ind.expand((6, 4))
        torch.manual_seed(2024)
        sample = expanded.sample((2,))
        self.assertEqual(sample.shape, torch.Size([2, 6, 4, 3]))
        self.assertTrue(torch.isfinite(sample).all())

    def test_expand_invalid_raises(self):
        """目标 batch_shape 与基分布参数不可广播时必须抛 RuntimeError。"""
        base = torch.distributions.Normal(
            torch.zeros(4, 3, device=self.device),
            torch.ones(4, 3, device=self.device))
        ind = torch.distributions.Independent(base, 1)
        with self.assertRaises(RuntimeError):
            ind.expand((2,))


if __name__ == "__main__":
    run_tests()
