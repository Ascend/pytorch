# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.log_normal.LogNormal.base_dist 接口功能正确性
API 名称：torch.distributions.log_normal.LogNormal.base_dist
API 签名：base_dist 实例属性，类型为 Normal
         （上游源码 torch/distributions/log_normal.py L46-L47：base_dist =
          Normal(loc, scale)，再经 ExpTransform 变换 Y=exp(X)）

覆盖维度表：
| 覆盖维度     | 说明                                      | 覆盖情况                               |
|--------------|-------------------------------------------|----------------------------------------|
| 空/非空      | 标量参数空 batch；batched 参数非空 batch  | 已覆盖（test_base_dist_type_and_params、|
|              | base_dist.batch_shape 均正确              | test_base_dist_batch_shape）           |
| 枚举选项     | 本接口无枚举入参                          | 不适用                                 |
| 参数类型     | loc/scale 传 float vs Tensor（含 batched）| 已覆盖（test_base_dist_type_and_params、|
|              |                                           | test_base_dist_batch_shape）           |
| 传参与不传参 | 无参属性，仅默认读取                      | 已覆盖（各用例均直接读属性）           |
| 等价类/边界  | y=1（log(y)=0，变换不动点）边界上的变换   | 已覆盖（test_base_dist_log_prob_       |
|              | 关系                                      | relation）                             |
| 正常传参     | base_dist 类型为 Normal，loc/scale 与构造 | 已覆盖（test_base_dist_type_and_params、|
|              | 参数一致，log_prob 与正态闭式解及         | test_base_dist_log_prob_closed_form、  |
|              | Y=exp(X) 变换关系一致                     | test_base_dist_log_prob_relation）     |
| 异常传参     | 无入参属性，不存在异常传参路径            | 不适用                                 |
| 混合设备类型 | 纯 Python 契约测试，张量统一经 self.device| 不适用（属性语义与设备无关）           |
|              | 构造，不涉及跨设备混用                    |                                        |

未覆盖项及原因：
- 未覆盖：validate_args=True 时构造非法参数的行为。该路径属于构造器
  （__init__）的参数校验契约而非 base_dist 属性契约，不在本文件范围内伪对照。

注意：base_dist 是 LogNormal 的指数变换基分布，断言全部落在「返回类型
     isinstance(Normal)、参数一致性、正态闭式解、变换关系
     log p_Y(y)=log p_X(log y)-log(y)」四类可证伪契约上，浮点对照一律使用
     torch.testing.assert_close。
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
class TestLogNormalBaseDist(TestCase):
    """Test cases for torch.distributions.log_normal.LogNormal.base_dist."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_base_dist_type_and_params(self):
        """base_dist 是 Normal 实例：loc/scale 与构造参数一致（float 标量构造）。"""
        dist = torch.distributions.LogNormal(0.5, 2.0)
        base = dist.base_dist
        self.assertIsInstance(base, torch.distributions.Normal)
        # Python-float 参数经 torch.as_tensor 落在默认(CPU)设备，期望值同取 CPU。
        torch.testing.assert_close(base.loc, torch.full((), 0.5))
        torch.testing.assert_close(base.scale, torch.full((), 2.0))

    def test_base_dist_log_prob_closed_form(self):
        """base_dist.log_prob 与正态闭式解 -log(scale)-(x-loc)^2/(2*scale^2)-
        log(sqrt(2*pi)) 逐元素一致。"""
        loc = torch.tensor([0.5, -1.0], device=self.device)
        scale = torch.tensor([1.5, 2.0], device=self.device)
        dist = torch.distributions.LogNormal(loc, scale)
        base = dist.base_dist
        x = torch.tensor([[-1.0, 0.0], [0.5, 2.0]], device=self.device)
        expected = (-math.log(math.sqrt(2 * math.pi)) - scale.log()
                    - (x - loc).pow(2) / (2 * scale.pow(2)))
        torch.testing.assert_close(base.log_prob(x), expected)

    def test_base_dist_log_prob_relation(self):
        """变换关系：log p_Y(y) = log p_X(log(y)) - log(y)；y=1 处 log(y)=0 精确成立。"""
        loc = torch.tensor([0.5, -1.0], device=self.device)
        scale = torch.tensor([1.5, 2.0], device=self.device)
        dist = torch.distributions.LogNormal(loc, scale)
        base = dist.base_dist
        y = torch.tensor([[0.5, 1.0], [2.0, 4.0]], device=self.device)
        expected = base.log_prob(y.log()) - y.log()
        torch.testing.assert_close(dist.log_prob(y), expected)

    def test_base_dist_batch_shape(self):
        """batched 参数：base_dist 的 batch_shape/loc/scale 形状与构造参数一致。"""
        loc = torch.tensor([[0.5, -1.0], [0.0, 1.0]], device=self.device)
        scale = torch.tensor([[1.5, 2.0], [0.5, 1.0]], device=self.device)
        dist = torch.distributions.LogNormal(loc, scale)
        base = dist.base_dist
        self.assertEqual(base.batch_shape, torch.Size([2, 2]))
        self.assertEqual(dist.batch_shape, base.batch_shape)
        torch.testing.assert_close(base.loc, loc)
        torch.testing.assert_close(base.scale, scale)


if __name__ == "__main__":
    run_tests()
