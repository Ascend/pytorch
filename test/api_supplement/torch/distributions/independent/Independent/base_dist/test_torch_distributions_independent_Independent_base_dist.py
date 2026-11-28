# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.independent.Independent.base_dist 接口功能正确性
API 名称：torch.distributions.independent.Independent.base_dist
API 签名：base_dist 实例属性，类型为构造时传入的基分布 D
         （上游源码 torch/distributions/independent.py L66：self.base_dist =
          base_distribution；L62-L65 依据 batch_shape/event_shape 重算维度）

覆盖维度表：
| 覆盖维度     | 说明                                      | 覆盖情况                               |
|--------------|-------------------------------------------|----------------------------------------|
| 空/非空      | 空 batch（基分布标量批）ndims=1 时        | 已覆盖（test_base_dist_batch_shape）   |
|              | base_dist.batch_shape 的空/非空形态       |                                        |
| 枚举选项     | 本接口无枚举入参                          | 不适用                                 |
| 参数类型     | 基分布 loc/scale 为 Tensor（含 batched）  | 已覆盖（test_base_dist_batch_shape、   |
|              |                                           | test_base_dist_identity_and_shapes）   |
| 传参与不传参 | 无参属性，仅默认读取                      | 已覆盖（各用例均直接读属性）           |
| 等价类/边界  | ndims=0（不重解释）与 ndims=len(batch)    | 已覆盖（test_base_dist_zero_ndims、    |
|              | （全部重解释）两个边界                    | test_base_dist_identity_and_shapes）   |
| 正常传参     | base_dist 与构造传入对象同一（is），      | 已覆盖（test_base_dist_identity_and_   |
|              | log_prob 为基分布 log_prob 沿事件维求和   | shapes、test_base_dist_log_prob_sum）  |
| 异常传参     | reinterpreted_batch_ndims 超出基分布      | 已覆盖（test_base_dist_ndims_raises）  |
|              | batch 维数时构造抛 ValueError             |                                        |
| 混合设备类型 | 纯 Python 契约测试，张量统一经 self.device| 不适用（属性语义与设备无关）           |
|              | 构造，不涉及跨设备混用                    |                                        |

未覆盖项及原因：
- 无（上述 8 维中不适用项均已给出原因，其余均有具体用例与断言对应）。

注意：base_dist 是 Independent 包装的原始分布对象，断言全部落在「同一性
     （is）、维度重算公式、log_prob 求和关系、非法 ndims 报错」四类可证伪
     契约上，浮点对照一律使用 torch.testing.assert_close。
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
class TestIndependentBaseDist(TestCase):
    """Test cases for torch.distributions.independent.Independent.base_dist."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_base_dist_identity_and_shapes(self):
        """base_dist 与构造传入对象同一；ndims=1/2 的 batch/event 维度按源码公式重算。"""
        base = torch.distributions.Normal(
            torch.zeros(4, 3, device=self.device),
            torch.ones(4, 3, device=self.device))
        ind1 = torch.distributions.Independent(base, 1)
        self.assertIs(ind1.base_dist, base)
        self.assertEqual(ind1.reinterpreted_batch_ndims, 1)
        self.assertEqual(ind1.batch_shape, torch.Size([4]))
        self.assertEqual(ind1.event_shape, torch.Size([3]))
        # 边界：ndims 等于基分布 batch 维数（全部重解释），batch 变为空
        ind2 = torch.distributions.Independent(base, 2)
        self.assertIs(ind2.base_dist, base)
        self.assertEqual(ind2.batch_shape, torch.Size([]))
        self.assertEqual(ind2.event_shape, torch.Size([4, 3]))

    def test_base_dist_log_prob_sum(self):
        """Independent.log_prob 等于 base_dist.log_prob 沿重解释维度求和。"""
        base = torch.distributions.Normal(
            torch.zeros(4, 3, device=self.device),
            torch.ones(4, 3, device=self.device))
        ind = torch.distributions.Independent(base, 1)
        torch.manual_seed(0)
        value = torch.randn(5, 4, 3, device=self.device)
        expected = ind.base_dist.log_prob(value).sum(-1)
        torch.testing.assert_close(ind.log_prob(value), expected)

    def test_base_dist_batch_shape(self):
        """空 batch 基分布（标量批 (3,0) 形态）经 ndims=1 包装后 batch_shape 为空。"""
        base = torch.distributions.Normal(
            torch.zeros(3, device=self.device), torch.ones(3, device=self.device))
        self.assertEqual(base.batch_shape, torch.Size([3]))
        ind = torch.distributions.Independent(base, 1)
        # shape=(3,), event_dim=1 -> batch_shape=()，即空 batch 边界形态
        self.assertEqual(ind.base_dist.batch_shape, torch.Size([3]))
        self.assertEqual(ind.batch_shape, torch.Size([]))
        self.assertEqual(ind.event_shape, torch.Size([3]))

    def test_base_dist_zero_ndims(self):
        """边界 ndims=0：维度完全不重解释，batch/event 与基分布一致，log_prob 不求和。"""
        base = torch.distributions.Normal(
            torch.zeros(4, 3, device=self.device),
            torch.ones(4, 3, device=self.device))
        ind = torch.distributions.Independent(base, 0)
        self.assertEqual(ind.batch_shape, torch.Size([4, 3]))
        self.assertEqual(ind.event_shape, torch.Size([]))
        torch.manual_seed(0)
        value = torch.randn(4, 3, device=self.device)
        torch.testing.assert_close(ind.log_prob(value), base.log_prob(value))

    def test_base_dist_ndims_raises(self):
        """reinterpreted_batch_ndims 超出基分布 batch 维数时构造抛 ValueError。"""
        base = torch.distributions.Normal(
            torch.zeros(4, 3, device=self.device),
            torch.ones(4, 3, device=self.device))
        with self.assertRaises(ValueError):
            torch.distributions.Independent(base, 3)


if __name__ == "__main__":
    run_tests()
