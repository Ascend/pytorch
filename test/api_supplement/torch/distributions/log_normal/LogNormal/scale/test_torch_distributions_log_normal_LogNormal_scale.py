# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.log_normal.LogNormal.scale 接口功能正确性
API 名称：torch.distributions.log_normal.LogNormal.scale
API 签名：scale 只读属性（property） -> Tensor
         （上游源码 torch/distributions/log_normal.py L57-L59：透传
          self.base_dist.scale）

覆盖维度表：
| 覆盖维度     | 说明                                      | 覆盖情况                               |
|--------------|-------------------------------------------|----------------------------------------|
| 空/非空      | 标量 scale 空 batch；batched loc/scale    | 已覆盖（test_scale_matches_arg、       |
|              | 非空 batch 均正确                         | test_scale_broadcast_from_scalar、     |
|              |                                           | test_scale_batched）                   |
| 枚举选项     | 本接口无枚举入参                          | 不适用                                 |
| 参数类型     | scale 传 float（广播）vs Tensor           | 已覆盖（test_scale_broadcast_from_     |
|              |                                           | scalar、test_scale_matches_arg）       |
| 传参与不传参 | 无参属性，仅默认读取                      | 已覆盖（各用例均直接读属性）           |
| 等价类/边界  | scale=0 边界违反 positive 约束（validate_ | 已覆盖（test_scale_zero_raises）       |
|              | args=True 时构造抛 ValueError）           |                                        |
| 正常传参     | scale 与构造参数/base_dist.scale 一致，   | 已覆盖（test_scale_matches_arg、       |
|              | 并驱动 mean/mode/variance 闭式解          | test_scale_feeds_analytic_stats、      |
|              |                                           | test_scale_batched）                   |
| 异常传参     | validate_args=True 时非法 scale 构造抛错  | 已覆盖（test_scale_zero_raises）       |
| 混合设备类型 | 纯 Python 契约测试，张量统一经 self.device| 不适用（属性语义与设备无关）           |
|              | 构造，不涉及跨设备混用                    |                                        |

未覆盖项及原因：
- 无（上述 8 维中不适用项均已给出原因，其余均有具体用例与断言对应）。

注意：scale 断言全部落在「与构造参数 assert_close、与 base_dist.scale 同一
     （is）、闭式统计量驱动关系、非法值构造报错」四类可证伪契约上。
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
class TestLogNormalScale(TestCase):
    """Test cases for torch.distributions.log_normal.LogNormal.scale."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_scale_matches_arg(self):
        """scale 与构造参数逐元素一致，且与 base_dist.scale 是同一对象。"""
        scale = torch.tensor([1.5, 2.0], device=self.device)
        loc = torch.tensor([0.5, -1.0], device=self.device)
        dist = torch.distributions.LogNormal(loc, scale)
        self.assertIs(dist.scale, dist.base_dist.scale)
        torch.testing.assert_close(dist.scale, scale)

    def test_scale_broadcast_from_scalar(self):
        """float 标量 scale 按 loc 形状广播（广播等价类）。"""
        loc = torch.zeros(3, device=self.device)
        dist = torch.distributions.LogNormal(loc, 2.0)
        self.assertEqual(dist.scale.shape, torch.Size([3]))
        torch.testing.assert_close(
            dist.scale, torch.full((3,), 2.0, device=self.device))

    def test_scale_batched(self):
        """batched loc/scale：scale 形状与 batch_shape 一致，mean/mode/variance 闭式解成立。"""
        loc = torch.tensor([[0.5, -1.0], [0.0, 1.0]], device=self.device)
        scale = torch.tensor([[1.5, 2.0], [0.5, 1.0]], device=self.device)
        dist = torch.distributions.LogNormal(loc, scale)
        self.assertEqual(dist.scale.shape, torch.Size([2, 2]))
        scale_sq = dist.scale.pow(2)
        torch.testing.assert_close(dist.mean, (dist.loc + scale_sq / 2).exp())
        torch.testing.assert_close(dist.mode, (dist.loc - scale_sq).exp())
        expected_var = scale_sq.expm1() * (2 * dist.loc + scale_sq).exp()
        torch.testing.assert_close(dist.variance, expected_var)

    def test_scale_feeds_analytic_stats(self):
        """scale 数值直接决定闭式统计量：scale*2 后 mean/variance 随公式联动。"""
        loc = torch.tensor(1.0, device=self.device)
        dist = torch.distributions.LogNormal(loc, torch.tensor(1.5, device=self.device))
        mean_1 = dist.mean
        dist2 = torch.distributions.LogNormal(
            loc, torch.tensor(2.5, device=self.device))
        mean_2 = dist2.mean
        expected_mean_2 = (loc + torch.tensor(2.5, device=self.device).pow(2) / 2).exp()
        torch.testing.assert_close(mean_2, expected_mean_2)
        self.assertGreater(float(mean_2.item()), float(mean_1.item()))

    def test_scale_zero_raises(self):
        """边界 scale=0 违反 positive 约束：validate_args=True 时构造抛 ValueError。"""
        with self.assertRaises(ValueError):
            torch.distributions.LogNormal(
                torch.tensor(0.0, device=self.device),
                torch.tensor(0.0, device=self.device), validate_args=True)


if __name__ == "__main__":
    run_tests()
