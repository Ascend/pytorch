# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.mixture_same_family.MixtureSameFamily.expand
         接口功能正确性
API 名称：torch.distributions.mixture_same_family.MixtureSameFamily.expand
API 签名：expand(self, batch_shape, _instance=None) -> MixtureSameFamily
         （上游源码 torch/distributions/mixture_same_family.py L111-L126：成分
          分布展开到 batch_shape+(k,)，混合分布展开到 batch_shape）

覆盖维度表：
| 覆盖维度     | 说明                                      | 覆盖情况                               |
|--------------|-------------------------------------------|----------------------------------------|
| 空/非空      | 空 batch（batch_shape=()）expand 到非空   | 已覆盖（test_expand_empty_to_batch）   |
|              | (4,)                                      |                                        |
| 枚举选项     | 本接口无枚举入参（batch_shape 为 Size）   | 不适用                                 |
| 参数类型     | mixture_probs/comp 参数均为 Tensor        | 已覆盖（test_expand_batch_shape_and_   |
|              |                                           | components、test_expand_log_prob）     |
| 传参与不传参 | 仅传 batch_shape，_instance 走默认 None   | 已覆盖（各用例均未传 _instance）       |
| 等价类/边界  | 追加前导维 (3,)→(6,3)；标量批 ()→(4,)；   | 已覆盖（test_expand_batch_shape_and_   |
|              | 非法收缩 (3,)→(2,)                        | components、test_expand_invalid_raises)|
| 正常传参     | expand 后 log_prob 与原分布逐元素一致，   | 已覆盖（test_expand_log_prob）         |
|              | 成分/混合子分布形状按源码公式重组         |                                        |
| 异常传参     | 不可广播的目标 batch_shape 抛 RuntimeError| 已覆盖（test_expand_invalid_raises）   |
| 混合设备类型 | 纯 Python 契约测试，张量统一经 self.device| 不适用（expand 语义与设备无关）        |
|              | 构造，不涉及跨设备混用                    |                                        |

未覆盖项及原因：
- 未覆盖：_instance 非默认路径。该参数是框架内部复用实例的私有协议
  （_get_checked_instance），公开契约上与默认路径等价，不做伪对照。

注意：expand 不改变成分数 k 与 event 维，断言全部落在「batch_shape 与两个
     子分布形状、log_prob 一致性、标量批边界、采样形状、非法 shape 报错」
     五类可证伪契约上，浮点对照一律使用 torch.testing.assert_close。
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
class TestMixtureSameFamilyExpand(TestCase):
    """Test cases for torch.distributions.mixture_same_family.MixtureSameFamily.expand."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_dist(self):
        probs = torch.full((3, 5), 0.2, device=self.device)
        mix = torch.distributions.Categorical(probs)
        comp = torch.distributions.Normal(
            torch.randn(3, 5, device=self.device), torch.rand(3, 5, device=self.device) + 0.5)
        return torch.distributions.MixtureSameFamily(mix, comp)

    def test_expand_batch_shape_and_components(self):
        """expand 后 batch_shape 正确，成分/混合子分布按源码公式重组形状。"""
        dist = self._make_dist()
        self.assertEqual(dist.batch_shape, torch.Size([3]))
        expanded = dist.expand((6, 3))
        self.assertEqual(expanded.batch_shape, torch.Size([6, 3]))
        self.assertEqual(expanded.event_shape, torch.Size([]))
        self.assertEqual(expanded._num_component, 5)
        # 源码公式：成分展开到 batch_shape+(k,)，混合展开到 batch_shape
        self.assertEqual(expanded.component_distribution.batch_shape, torch.Size([6, 3, 5]))
        self.assertEqual(expanded.mixture_distribution.batch_shape, torch.Size([6, 3]))

    def test_expand_log_prob(self):
        """expand 后 log_prob 与原分布逐元素一致（原分布对 value 广播）。"""
        dist = self._make_dist()
        expanded = dist.expand((6, 3))
        torch.manual_seed(0)
        value = torch.randn(6, 3, device=self.device)
        torch.testing.assert_close(expanded.log_prob(value), dist.log_prob(value))
        self.assertEqual(expanded.log_prob(value).shape, torch.Size([6, 3]))

    def test_expand_empty_to_batch(self):
        """空 batch（标量混合批）expand 到非空 batch (4,)。"""
        probs = torch.full((5,), 0.2, device=self.device)
        mix = torch.distributions.Categorical(probs)
        comp = torch.distributions.Normal(
            torch.randn(5, device=self.device), torch.rand(5, device=self.device) + 0.5)
        dist = torch.distributions.MixtureSameFamily(mix, comp)
        self.assertEqual(dist.batch_shape, torch.Size([]))
        expanded = dist.expand((4,))
        self.assertEqual(expanded.batch_shape, torch.Size([4]))
        self.assertEqual(expanded.component_distribution.batch_shape, torch.Size([4, 5]))

    def test_expand_event_dims_case(self):
        """多 event 维成分（Independent(Normal,1)）：expand 后 event 维保持。"""
        probs = torch.full((3, 5), 0.2, device=self.device)
        mix = torch.distributions.Categorical(probs)
        comp = torch.distributions.Independent(
            torch.distributions.Normal(
                torch.randn(3, 5, 2, device=self.device),
                torch.rand(3, 5, 2, device=self.device) + 0.5), 1)
        dist = torch.distributions.MixtureSameFamily(mix, comp)
        self.assertEqual(dist.event_shape, torch.Size([2]))
        expanded = dist.expand((6, 3))
        self.assertEqual(expanded.batch_shape, torch.Size([6, 3]))
        self.assertEqual(expanded.event_shape, torch.Size([2]))

    def test_expand_seed_sample_shape(self):
        """固定种子采样：expand 后 sample 形状为 sample_shape + expanded batch + event。"""
        dist = self._make_dist()
        expanded = dist.expand((6, 3))
        torch.manual_seed(2024)
        sample = expanded.sample((4,))
        self.assertEqual(sample.shape, torch.Size([4, 6, 3]))
        self.assertTrue(torch.isfinite(sample).all())

    def test_expand_invalid_raises(self):
        """目标 batch_shape 与成分分布不可广播时必须抛 RuntimeError。"""
        dist = self._make_dist()
        with self.assertRaises(RuntimeError):
            dist.expand((2,))


if __name__ == "__main__":
    run_tests()
