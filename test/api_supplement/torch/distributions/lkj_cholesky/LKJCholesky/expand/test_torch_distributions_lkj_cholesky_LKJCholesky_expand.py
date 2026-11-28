# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.lkj_cholesky.LKJCholesky.expand 接口功能正确性
API 名称：torch.distributions.lkj_cholesky.LKJCholesky.expand
API 签名：expand(self, batch_shape, _instance=None) -> LKJCholesky
         （上游源码 torch/distributions/lkj_cholesky.py L93-L103：展开
          concentration 与内部 Beta 辅助分布后以 batch_shape 重建）

覆盖维度表：
| 覆盖维度     | 说明                                      | 覆盖情况                               |
|--------------|-------------------------------------------|----------------------------------------|
| 空/非空      | 空 batch（标量 concentration）expand 到   | 已覆盖（test_expand_scalar_to_batch）  |
|              | 非空 (4,)                                 |                                        |
| 枚举选项     | 本接口无枚举入参（batch_shape 为 Size）   | 不适用                                 |
| 参数类型     | concentration 传 float vs Tensor          | 已覆盖（test_expand_scalar_to_batch、  |
|              |                                           | test_expand_batch_shape_and_params）   |
| 传参与不传参 | 仅传 batch_shape，_instance 走默认 None   | 已覆盖（各用例均未传 _instance）       |
| 等价类/边界  | 追加前导维 (2,)→(5,2)；恒等 (2,)→(2,)；   | 已覆盖（test_expand_batch_shape_and_   |
|              | 非法收缩 (2,)→(1,)                        | params、test_expand_invalid_raises）   |
| 正常传参     | expand 后 log_prob 与原分布逐元素一致；   | 已覆盖（test_expand_log_prob、         |
|              | 采样行范数=1 的相关阵 Cholesky 性质成立   | test_expand_seed_sample_shape）        |
| 异常传参     | 不可广播的目标 batch_shape 抛 RuntimeError| 已覆盖（test_expand_invalid_raises）   |
| 混合设备类型 | 纯 Python 契约测试，张量统一经 self.device| 不适用（expand 语义与设备无关）        |
|              | 构造，不涉及跨设备混用                    |                                        |

未覆盖项及原因：
- 未覆盖：_instance 非默认路径。该参数是框架内部复用实例的私有协议
  （_get_checked_instance），公开契约上与默认路径等价，不做伪对照。

注意：expand 不改变 dim/event_shape 与采样语义，断言全部落在「batch_shape/
     concentration 形状、log_prob 一致性、采样形状与行范数=1（相关阵
     Cholesky 定义）、非法 shape 报错」五类可证伪契约上，浮点对照一律使用
     torch.testing.assert_close。
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


def _corr_cholesky_factor(shape, dim, device):
    """构造 shape 批的固定相关阵（非对角 0.3）的 Cholesky 因子作为合法样本值。

    Cholesky 分解在 CPU 上完成（NPU aclnnLinalgCholesky 对该输入当前报错，
    已在 UT 报告缺陷登记），随后将合法因子移到目标设备——这只是测试脚手架，
    不影响被测的 LKJCholesky.expand 契约本身。
    """
    corr = torch.full(shape + (dim, dim), 0.3)
    idx = torch.arange(dim)
    corr[..., idx, idx] = 1.0
    return torch.linalg.cholesky(corr).to(device)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestLKJCholeskyExpand(TestCase):
    """Test cases for torch.distributions.lkj_cholesky.LKJCholesky.expand."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_expand_batch_shape_and_params(self):
        """expand 追加前导维后 batch_shape/event_shape/concentration 均正确。"""
        concentration = torch.tensor([0.5, 1.5], device=self.device)
        dist = torch.distributions.LKJCholesky(3, concentration)
        expanded = dist.expand((5, 2))
        self.assertEqual(expanded.batch_shape, torch.Size([5, 2]))
        self.assertEqual(expanded.event_shape, torch.Size([3, 3]))
        self.assertEqual(expanded.dim, 3)
        self.assertEqual(expanded.concentration.shape, torch.Size([5, 2]))
        torch.testing.assert_close(expanded.concentration, concentration.expand((5, 2)))
        # 恒等边界：expand 到原 batch 形状后 concentration 保持逐元素一致
        same = dist.expand((2,))
        torch.testing.assert_close(same.concentration, concentration)

    def test_expand_log_prob(self):
        """expand 后 log_prob 与原分布逐元素一致（原分布对合法 Cholesky 因子广播）。"""
        concentration = torch.tensor([0.5, 1.5], device=self.device)
        dist = torch.distributions.LKJCholesky(3, concentration)
        expanded = dist.expand((5, 2))
        value = _corr_cholesky_factor((5, 2), 3, self.device)
        torch.testing.assert_close(expanded.log_prob(value), dist.log_prob(value))
        # 可证伪锚点：log_prob 形状必须为 expanded batch_shape
        self.assertEqual(expanded.log_prob(value).shape, torch.Size([5, 2]))

    def test_expand_seed_sample_shape(self):
        """固定种子采样：形状正确且样本满足相关阵 Cholesky 的行范数=1 定义。"""
        concentration = torch.tensor([0.5, 1.5], device=self.device)
        dist = torch.distributions.LKJCholesky(3, concentration)
        expanded = dist.expand((5, 2))
        torch.manual_seed(2024)
        sample = expanded.sample((4,))
        self.assertEqual(sample.shape, torch.Size([4, 5, 2, 3, 3]))
        row_norms = sample.norm(dim=-1)
        torch.testing.assert_close(
            row_norms, torch.ones_like(row_norms), check_dtype=False)

    def test_expand_scalar_to_batch(self):
        """空 batch（float 标量 concentration）expand 到非空 batch (4,)。"""
        dist = torch.distributions.LKJCholesky(3, 1.0)
        self.assertEqual(dist.batch_shape, torch.Size([]))
        expanded = dist.expand((4,))
        self.assertEqual(expanded.batch_shape, torch.Size([4]))
        self.assertEqual(expanded.event_shape, torch.Size([3, 3]))
        # Python-float concentration 落在默认(CPU)设备，期望取 CPU 张量。
        torch.testing.assert_close(
            expanded.concentration, torch.full((4,), 1.0))

    def test_expand_invalid_raises(self):
        """目标 batch_shape 与现有 concentration 不可广播时必须抛 RuntimeError。"""
        concentration = torch.tensor([0.5, 1.5], device=self.device)
        dist = torch.distributions.LKJCholesky(3, concentration)
        with self.assertRaises(RuntimeError):
            dist.expand((1,))


if __name__ == "__main__":
    run_tests()
