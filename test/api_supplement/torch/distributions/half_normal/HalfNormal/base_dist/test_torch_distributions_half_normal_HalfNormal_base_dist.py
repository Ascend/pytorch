# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.half_normal.HalfNormal.base_dist 接口功能正确性
API 名称：torch.distributions.half_normal.HalfNormal.base_dist
API 签名：base_dist 实例属性，类型为 Normal
         （上游源码 torch/distributions/half_normal.py L45-L46：base_dist =
          Normal(0, scale, validate_args=False)，再经 AbsTransform 变换）

覆盖维度表：
| 覆盖维度     | 说明                                      | 覆盖情况                               |
|--------------|-------------------------------------------|----------------------------------------|
| 空/非空      | 标量 scale 空 batch；batched scale 非空   | 已覆盖（test_base_dist_type_and_scale、|
|              | batch，base_dist.batch_shape 均正确       | test_base_dist_batch_shape）           |
| 枚举选项     | 本接口无枚举入参                          | 不适用                                 |
| 参数类型     | scale 传 float 标量 vs Tensor             | 已覆盖（test_base_dist_type_and_scale、|
|              |                                           | test_base_dist_batch_shape）           |
| 传参与不传参 | 无参属性，仅默认读取                      | 已覆盖（各用例均直接读属性）           |
| 等价类/边界  | x=0 与 x<0 两个 support 边界上            | 已覆盖（test_base_dist_log_prob_       |
|              | HalfNormal.log_prob 与 base_dist 的关系   | relation_to_half_normal）              |
| 正常传参     | base_dist 类型为 Normal，loc/scale 与构造 | 已覆盖（test_base_dist_type_and_scale、|
|              | 参数一致，log_prob 与正态闭式解一致       | test_base_dist_log_prob_closed_form）  |
| 异常传参     | 无入参属性，不存在异常传参路径            | 不适用                                 |
| 混合设备类型 | 纯 Python 契约测试，张量统一经 self.device| 不适用（属性语义与设备无关）           |
|              | 构造，不涉及跨设备混用                    |                                        |

未覆盖项及原因：
- 未覆盖：validate_args=True 时构造非法 scale 的行为。该路径属于构造器
  （__init__）的参数校验契约而非 base_dist 属性契约，不在本文件范围内伪对照。

注意：base_dist 是 HalfNormal 的绝对值变换基分布，断言全部落在「返回类型
     isinstance(Normal)、参数一致性、log_prob 闭式解、x=0/x<0 边界关系」
     四类可证伪契约上，浮点对照一律使用 torch.testing.assert_close。
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
class TestHalfNormalBaseDist(TestCase):
    """Test cases for torch.distributions.half_normal.HalfNormal.base_dist."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_base_dist_type_and_scale(self):
        """base_dist 是 Normal 实例：scale 与构造参数一致，loc 为 0（float 标量构造）。"""
        dist = torch.distributions.HalfNormal(2.0)
        base = dist.base_dist
        self.assertIsInstance(base, torch.distributions.Normal)
        # Python-float 参数落在默认(CPU)设备，期望值同取 CPU 张量。
        torch.testing.assert_close(base.scale, torch.full((), 2.0))
        torch.testing.assert_close(base.loc, torch.zeros(()))
        # HalfNormal.scale 透传 base_dist.scale（源码 L53-L54）
        torch.testing.assert_close(dist.scale, base.scale)

    def test_base_dist_log_prob_closed_form(self):
        """base_dist.log_prob 与正态闭式解 -log(scale)-x^2/(2*scale^2)-log(sqrt(2*pi)) 一致。"""
        scale = torch.tensor([1.5, 2.0], device=self.device)
        dist = torch.distributions.HalfNormal(scale)
        base = dist.base_dist
        x = torch.tensor([[0.5, 1.0], [3.0, 0.0]], device=self.device)
        expected = (-math.log(math.sqrt(2 * math.pi)) - scale.log()
                    - x.pow(2) / (2 * scale.pow(2)))
        torch.testing.assert_close(base.log_prob(x), expected)

    def test_base_dist_log_prob_relation_to_half_normal(self):
        """x>=0 时 HalfNormal.log_prob = base_dist.log_prob + log(2)；x<0 时为 -inf。"""
        scale = torch.tensor([1.5, 2.0], device=self.device)
        # validate_args=False: the x<0 branch deliberately feeds out-of-support
        # values to pin the -inf contract, which validate_args would reject.
        dist = torch.distributions.HalfNormal(scale, validate_args=False)
        base = dist.base_dist
        x_pos = torch.tensor([0.5, 1.0], device=self.device)
        torch.testing.assert_close(
            dist.log_prob(x_pos), base.log_prob(x_pos) + math.log(2))
        # 边界 x=0：与正数分支同规则
        x_zero = torch.zeros(2, device=self.device)
        torch.testing.assert_close(
            dist.log_prob(x_zero), base.log_prob(x_zero) + math.log(2))
        # 边界 x<0：support 外，log_prob 必须为 -inf，而 base_dist 为有限值
        x_neg = torch.tensor([-0.5, -1.0], device=self.device)
        self.assertTrue(torch.isinf(dist.log_prob(x_neg)).all())
        self.assertTrue(torch.isfinite(base.log_prob(x_neg)).all())

    def test_base_dist_batch_shape(self):
        """batched scale：base_dist 的 batch_shape/loc/scale 形状与构造参数一致。"""
        scale = torch.tensor([1.5, 2.0], device=self.device)
        dist = torch.distributions.HalfNormal(scale)
        base = dist.base_dist
        self.assertEqual(base.batch_shape, torch.Size([2]))
        self.assertEqual(base.loc.shape, torch.Size([2]))
        self.assertEqual(base.scale.shape, torch.Size([2]))
        torch.testing.assert_close(base.scale, scale)
        torch.testing.assert_close(base.loc, torch.zeros_like(scale))


if __name__ == "__main__":
    run_tests()
