# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.mixture_same_family.MixtureSameFamily.mean
         接口功能正确性
API 名称：torch.distributions.mixture_same_family.MixtureSameFamily.mean
API 签名：mean 只读属性（property） -> Tensor
         （上游源码 torch/distributions/mixture_same_family.py L141-L146：
          sum(probs * component.mean, dim=-1-event_ndims)）

覆盖维度表：
| 覆盖维度     | 说明                                      | 覆盖情况                               |
|--------------|-------------------------------------------|----------------------------------------|
| 空/非空      | 空 batch（标量批）、batched (3,) 与标量   | 已覆盖（test_mean_single_component、   |
|              | 批+event>1 的 mean 形状均正确             | test_mean_shape_batched、              |
|              |                                           | test_mean_multi_dim_event、            |
|              |                                           | test_mean_scalar_batch_multi_dim_event）|
| 枚举选项     | 本接口无枚举入参                          | 不适用                                 |
| 参数类型     | mixture probs/comp 参数均为 Tensor        | 已覆盖（test_mean_weighted_sum、       |
|              |                                           | test_mean_uniform_equals_average）     |
| 传参与不传参 | 无参属性，仅默认读取                      | 已覆盖（各用例均直接读属性）           |
| 等价类/边界  | k=1 单成分边界：mean 退化为成分均值       | 已覆盖（test_mean_single_component）   |
| 正常传参     | 非均匀权重下 mean=加权和（解析值 2.5）；  | 已覆盖（test_mean_weighted_sum、       |
|              | 均匀权重下 mean=成分均值                  | test_mean_uniform_equals_average、     |
|              |                                           | test_mean_multi_dim_event）            |
| 异常传参     | 无参属性，不存在异常传参路径              | 不适用                                 |
| 混合设备类型 | 纯 Python 契约测试，张量统一经 self.device| 不适用（属性语义与设备无关）           |
|              | 构造，不涉及跨设备混用                    |                                        |

未覆盖项及原因：
- 无（上述 8 维中不适用项均已给出原因，其余均有具体用例与断言对应）。

注意：mean 断言全部落在「与手算解析值对照（0.25*1+0.75*3=2.5）、均匀权重
     闭式解（成分均值）、event 维>1 的形状与解析对照、k=1 退化边界」四类
     可证伪契约上，浮点对照一律使用 torch.testing.assert_close。
     torch 2.15 契约（v2.15 mixture_same_family.py）：batch_shape 取
     component_distribution.batch_shape[:-1]，k 为成分维；mean 沿
     dim=-1-event_ndims 对 probs*component.mean 求和，标量混合批（k 折入
     成分批）时 batch_shape 为空、mean 形状为 event 形状。
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
class TestMixtureSameFamilyMean(TestCase):
    """Test cases for torch.distributions.mixture_same_family.MixtureSameFamily.mean."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_mean_weighted_sum(self):
        """非均匀权重：mean 必须等于手算加权和 0.25*1 + 0.75*3 = 2.5。"""
        probs = torch.tensor([[0.25, 0.75]], device=self.device)
        mix = torch.distributions.Categorical(probs)
        comp = torch.distributions.Normal(
            torch.tensor([[1.0, 3.0]], device=self.device),
            torch.ones(1, 2, device=self.device))
        dist = torch.distributions.MixtureSameFamily(mix, comp)
        self.assertEqual(dist.batch_shape, torch.Size([1]))
        torch.testing.assert_close(
            dist.mean, torch.tensor([2.5], device=self.device))

    def test_mean_uniform_equals_average(self):
        """均匀权重：mean 必须等于成分均值的算术平均（独立闭式口径）。"""
        probs = torch.full((3, 5), 0.2, device=self.device)
        mix = torch.distributions.Categorical(probs)
        torch.manual_seed(0)
        comp = torch.distributions.Normal(
            torch.randn(3, 5, device=self.device),
            torch.rand(3, 5, device=self.device) + 0.5)
        dist = torch.distributions.MixtureSameFamily(mix, comp)
        expected = dist.component_distribution.mean.mean(-1)
        torch.testing.assert_close(dist.mean, expected)
        self.assertEqual(dist.mean.shape, torch.Size([3]))

    def test_mean_multi_dim_event(self):
        """event 维>1（Independent(Normal,1)）：batched 混合下 mean 形状 (batch,event)。"""
        # torch 2.15 契约：batch_shape = component.batch_shape[:-1]，k 为成分维，
        # mean 沿 dim=-1-event_ndims（即成分维）求加权和。
        probs = torch.full((4, 3), 1.0 / 3.0, device=self.device)
        mix = torch.distributions.Categorical(probs)
        comp = torch.distributions.Independent(
            torch.distributions.Normal(
                torch.randn(4, 3, 2, device=self.device),
                torch.rand(4, 3, 2, device=self.device) + 0.5), 1)
        dist = torch.distributions.MixtureSameFamily(mix, comp)
        self.assertEqual(dist.batch_shape, torch.Size([4]))
        self.assertEqual(dist.event_shape, torch.Size([2]))
        self.assertEqual(dist.mean.shape, torch.Size([4, 2]))
        # 均匀权重闭式解：mean 沿成分维等于成分均值的算术平均
        expected = dist.component_distribution.mean.mean(1)
        torch.testing.assert_close(dist.mean, expected)

    def test_mean_scalar_batch_multi_dim_event(self):
        """标量混合批（k 折入成分批）+ event 维>1：batch 为空，mean 形状为 event。"""
        probs = torch.full((4,), 0.25, device=self.device)
        mix = torch.distributions.Categorical(probs)
        comp = torch.distributions.Independent(
            torch.distributions.Normal(
                torch.randn(4, 2, device=self.device),
                torch.rand(4, 2, device=self.device) + 0.5), 1)
        dist = torch.distributions.MixtureSameFamily(mix, comp)
        self.assertEqual(dist.batch_shape, torch.Size([]))
        self.assertEqual(dist.event_shape, torch.Size([2]))
        self.assertEqual(dist.mean.shape, torch.Size([2]))
        expected = dist.component_distribution.mean.mean(0)
        torch.testing.assert_close(dist.mean, expected)

    def test_mean_single_component(self):
        """k=1 退化边界：mean 与成分均值完全一致（batch/event 均为空，结果为标量张量）。"""
        mix = torch.distributions.Categorical(torch.ones(1, device=self.device))
        comp = torch.distributions.Normal(
            torch.tensor([2.0], device=self.device),
            torch.ones(1, device=self.device))
        dist = torch.distributions.MixtureSameFamily(mix, comp)
        self.assertEqual(dist.batch_shape, torch.Size([]))
        # dist.batch_shape=() 且 event=()：sum(dim=-1) 压缩后 mean 为 0 维标量
        self.assertEqual(dist.mean.shape, torch.Size([]))
        torch.testing.assert_close(dist.mean, torch.tensor(2.0, device=self.device))

    def test_mean_shape_batched(self):
        """batched (3,) 标量 event：mean 形状必须为 batch_shape。"""
        probs = torch.full((3, 5), 0.2, device=self.device)
        mix = torch.distributions.Categorical(probs)
        comp = torch.distributions.Normal(
            torch.randn(3, 5, device=self.device),
            torch.rand(3, 5, device=self.device) + 0.5)
        dist = torch.distributions.MixtureSameFamily(mix, comp)
        self.assertEqual(dist.mean.shape, dist.batch_shape)


if __name__ == "__main__":
    run_tests()
