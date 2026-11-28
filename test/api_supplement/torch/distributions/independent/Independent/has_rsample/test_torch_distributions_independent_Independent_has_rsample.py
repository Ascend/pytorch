# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.independent.Independent.has_rsample 接口功能正确性
API 名称：torch.distributions.independent.Independent.has_rsample
API 签名：has_rsample 布尔属性（property） -> bool
         （上游源码 torch/distributions/independent.py L84-L86：透传
          base_dist.has_rsample）

覆盖维度表：
| 覆盖维度     | 说明                                      | 覆盖情况                               |
|--------------|-------------------------------------------|----------------------------------------|
| 空/非空      | 无空值语义（布尔属性）                    | 不适用                                 |
| 枚举选项     | 属性取值仅 True/False 两种                | 已覆盖（test_true_with_normal_base、   |
|              |                                           | test_false_with_bernoulli_base）       |
| 参数类型     | 返回类型必须为 Python bool                | 已覆盖（test_type_is_bool）            |
| 传参与不传参 | 无参属性，仅默认读取                      | 已覆盖（各用例均直接读属性）           |
| 等价类/边界  | 基分布可/不可重参数化两类（Normal vs      | 已覆盖（test_true_with_normal_base、   |
|              | Bernoulli）                               | test_false_with_bernoulli_base）       |
| 正常传参     | 与 base_dist.has_rsample 逐例一致         | 已覆盖（test_matches_base_dist）       |
| 异常传参     | 无入参属性；has_rsample=False 时的行为    | 已覆盖（test_rsample_raises_when_      |
|              | 契约（rsample 抛 NotImplementedError）    | unsupported）                          |
| 混合设备类型 | 纯 Python 布尔契约，不涉及设备张量        | 不适用                                 |

未覆盖项及原因：
- 无（上述 8 维中不适用项均已给出原因，其余均有具体用例与断言对应）。

注意：布尔取值断言使用 assertIs 与源码布尔字面量严格对照；True 分支另以
     「rsample 结果带梯度」这一可证伪行为佐证，False 分支以 rsample 抛
     NotImplementedError 佐证。
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
class TestIndependentHasRsample(TestCase):
    """Test cases for torch.distributions.independent.Independent.has_rsample."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_type_is_bool(self):
        """has_rsample 的返回类型必须是 Python bool。"""
        base = torch.distributions.Normal(
            torch.zeros(3, device=self.device), torch.ones(3, device=self.device))
        ind = torch.distributions.Independent(base, 1)
        self.assertIsInstance(ind.has_rsample, bool)

    def test_true_with_normal_base(self):
        """基分布 Normal 可重参数化：has_rsample 必须为 True。"""
        base = torch.distributions.Normal(
            torch.zeros(3, device=self.device), torch.ones(3, device=self.device))
        ind = torch.distributions.Independent(base, 1)
        self.assertIs(ind.has_rsample, True)

    def test_false_with_bernoulli_base(self):
        """基分布 Bernoulli 不可重参数化：has_rsample 必须为 False。"""
        base = torch.distributions.Bernoulli(0.5)
        self.assertIs(base.has_rsample, False)
        ind = torch.distributions.Independent(base, 0)
        self.assertIs(ind.has_rsample, False)

    def test_matches_base_dist(self):
        """has_rsample 与 base_dist.has_rsample 逐例严格一致（identity 断言）。"""
        normal = torch.distributions.Normal(
            torch.zeros(3, device=self.device), torch.ones(3, device=self.device))
        bernoulli = torch.distributions.Bernoulli(0.5)
        ind_normal = torch.distributions.Independent(normal, 1)
        ind_bernoulli = torch.distributions.Independent(bernoulli, 0)
        self.assertIs(ind_normal.has_rsample, ind_normal.base_dist.has_rsample)
        self.assertIs(ind_bernoulli.has_rsample, ind_bernoulli.base_dist.has_rsample)

    def test_rsample_raises_when_unsupported(self):
        """行为后果：has_rsample=False 时调用 rsample 必须抛 NotImplementedError。"""
        base = torch.distributions.Bernoulli(0.5)
        ind = torch.distributions.Independent(base, 0)
        self.assertIs(ind.has_rsample, False)
        with self.assertRaises(NotImplementedError):
            ind.rsample()

    def test_rsample_has_grad_when_supported(self):
        """行为佐证：has_rsample=True 时 rsample 结果必须带梯度。"""
        loc = torch.zeros(3, device=self.device, requires_grad=True)
        scale = torch.ones(3, device=self.device, requires_grad=True)
        base = torch.distributions.Normal(loc, scale)
        ind = torch.distributions.Independent(base, 1)
        sample = ind.rsample()
        self.assertTrue(sample.requires_grad)
        self.assertIsNotNone(sample.grad_fn)


if __name__ == "__main__":
    run_tests()
