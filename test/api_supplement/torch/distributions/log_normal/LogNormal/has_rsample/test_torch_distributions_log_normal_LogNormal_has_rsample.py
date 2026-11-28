# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.log_normal.LogNormal.has_rsample 接口功能正确性
API 名称：torch.distributions.log_normal.LogNormal.has_rsample
API 签名：has_rsample 类级布尔属性 = True
         （上游源码 torch/distributions/log_normal.py L36：has_rsample = True）

覆盖维度表：
| 覆盖维度     | 说明                                      | 覆盖情况                               |
|--------------|-------------------------------------------|----------------------------------------|
| 空/非空      | 无空值语义（布尔属性）                    | 不适用                                 |
| 枚举选项     | 属性取值唯一（True）                      | 已覆盖（test_class_and_instance_true） |
| 参数类型     | 返回类型必须为 Python bool                | 已覆盖（test_class_and_instance_true） |
| 传参与不传参 | 无参类级属性，仅默认读取                  | 已覆盖（各用例均直接读属性）           |
| 等价类/边界  | 类级取值与实例级取值两个访问面            | 已覆盖（test_class_and_instance_true） |
| 正常传参     | 与源码声明（True）及 base_dist.has_rsample| 已覆盖（test_matches_source_and_base、 |
|              | 一致                                      | test_rsample_has_grad）                |
| 异常传参     | 无入参属性，不存在异常传参路径            | 不适用                                 |
| 混合设备类型 | 纯 Python 布尔契约，不涉及设备张量        | 不适用                                 |

未覆盖项及原因：
- 无（上述 8 维中不适用项均已给出原因，其余均有具体用例与断言对应）。

注意：布尔取值断言使用 assertIs 与源码布尔字面量严格对照；另以「rsample
     结果带梯度、sample 结果无梯度」这一可证伪行为佐证 True 的语义，而非
     仅做类型断言。
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
class TestLogNormalHasRsample(TestCase):
    """Test cases for torch.distributions.log_normal.LogNormal.has_rsample."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_class_and_instance_true(self):
        """类级与实例级访问均为 Python bool 的 True。"""
        self.assertIsInstance(torch.distributions.LogNormal.has_rsample, bool)
        self.assertIs(torch.distributions.LogNormal.has_rsample, True)
        dist = torch.distributions.LogNormal(0.0, 1.0)
        self.assertIs(dist.has_rsample, True)

    def test_matches_source_and_base(self):
        """与 base_dist.has_rsample 一致（TransformedDistribution 透传语义）。"""
        dist = torch.distributions.LogNormal(0.0, 1.0)
        self.assertIs(dist.has_rsample, dist.base_dist.has_rsample)
        self.assertIs(dist.base_dist.has_rsample, True)

    def test_rsample_has_grad(self):
        """行为佐证：参数 requires_grad 时 rsample 结果必须带梯度。"""
        loc = torch.zeros(3, device=self.device, requires_grad=True)
        scale = torch.ones(3, device=self.device, requires_grad=True)
        dist = torch.distributions.LogNormal(loc, scale)
        torch.manual_seed(2024)
        sample = dist.rsample()
        self.assertTrue(sample.requires_grad)
        self.assertIsNotNone(sample.grad_fn)

    def test_sample_has_no_grad(self):
        """对照行为：sample() 在 no_grad 下执行，即使参数 requires_grad 也无梯度。"""
        loc = torch.zeros(3, device=self.device, requires_grad=True)
        scale = torch.ones(3, device=self.device, requires_grad=True)
        dist = torch.distributions.LogNormal(loc, scale)
        torch.manual_seed(2024)
        sample = dist.sample()
        self.assertFalse(sample.requires_grad)
        self.assertIsNone(sample.grad_fn)


if __name__ == "__main__":
    run_tests()
