# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.mixture_same_family.MixtureSameFamily.
         has_rsample 接口功能正确性
API 名称：torch.distributions.mixture_same_family.MixtureSameFamily.has_rsample
API 签名：has_rsample 类级布尔属性 = False
         （上游源码 torch/distributions/mixture_same_family.py L57）

覆盖维度表：
| 覆盖维度     | 说明                                      | 覆盖情况                               |
|--------------|-------------------------------------------|----------------------------------------|
| 空/非空      | 无空值语义（布尔属性）                    | 不适用                                 |
| 枚举选项     | 属性取值唯一（False）                     | 已覆盖（test_class_and_instance_false）|
| 参数类型     | 返回类型必须为 Python bool                | 已覆盖（test_class_and_instance_false）|
| 传参与不传参 | 无参类级属性，仅默认读取                  | 已覆盖（各用例均直接读属性）           |
| 等价类/边界  | 类级取值与实例级取值两个访问面；与成分分  | 已覆盖（test_class_and_instance_false、|
|              | 布有 rsample 的对照边界                   | test_component_has_rsample_contrast）  |
| 正常传参     | 与源码声明（False）一致，并以 sample 结果 | 已覆盖（test_sample_has_no_grad）      |
|              | 无梯度佐证                                |                                        |
| 异常传参     | 无入参属性；has_rsample=False 时的行为    | 已覆盖（test_rsample_raises）          |
|              | 契约（rsample 抛 NotImplementedError）    |                                        |
| 混合设备类型 | 纯 Python 布尔契约，不涉及设备张量        | 不适用                                 |

未覆盖项及原因：
- 无（上述 8 维中不适用项均已给出原因，其余均有具体用例与断言对应）。

注意：布尔取值断言使用 assertIs 与源码布尔字面量严格对照；False 语义以
     「rsample 抛 NotImplementedError、sample 结果无梯度」两个可证伪行为
     佐证，而非仅做类型断言。
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
class TestMixtureSameFamilyHasRsample(TestCase):
    """Test cases for MixtureSameFamily.has_rsample."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def _make_dist(self):
        logits = torch.randn(3, 5, device=self.device, requires_grad=True)
        mix = torch.distributions.Categorical(logits=logits)
        comp = torch.distributions.Normal(
            torch.randn(3, 5, device=self.device, requires_grad=True),
            torch.rand(3, 5, device=self.device, requires_grad=True) + 0.5)
        return torch.distributions.MixtureSameFamily(mix, comp)

    def test_class_and_instance_false(self):
        """类级与实例级访问均为 Python bool 的 False。"""
        self.assertIsInstance(torch.distributions.MixtureSameFamily.has_rsample, bool)
        self.assertIs(torch.distributions.MixtureSameFamily.has_rsample, False)
        dist = self._make_dist()
        self.assertIs(dist.has_rsample, False)

    def test_rsample_raises(self):
        """行为后果：has_rsample=False 时调用 rsample 必须抛 NotImplementedError。"""
        dist = self._make_dist()
        self.assertIs(dist.has_rsample, False)
        with self.assertRaises(NotImplementedError):
            dist.rsample()

    def test_sample_has_no_grad(self):
        """行为佐证：参数 requires_grad 时 sample() 结果也必须无梯度。"""
        dist = self._make_dist()
        torch.manual_seed(2024)
        sample = dist.sample((2,))
        self.assertFalse(sample.requires_grad)
        self.assertIsNone(sample.grad_fn)

    def test_component_has_rsample_contrast(self):
        """对照边界：成分分布（Normal）可重参数化，但混合分布本身恒为 False。"""
        dist = self._make_dist()
        self.assertIs(dist.component_distribution.has_rsample, True)
        self.assertIs(dist.has_rsample, False)


if __name__ == "__main__":
    run_tests()
