# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.independent.Independent.has_enumerate_support
         接口功能正确性
API 名称：torch.distributions.independent.Independent.has_enumerate_support
API 签名：has_enumerate_support 布尔属性（property） -> bool
         （上游源码 torch/distributions/independent.py L88-L92：ndims>0 时恒为
          False，否则透传 base_dist.has_enumerate_support）

覆盖维度表：
| 覆盖维度     | 说明                                      | 覆盖情况                               |
|--------------|-------------------------------------------|----------------------------------------|
| 空/非空      | 无空值语义（布尔属性）                    | 不适用                                 |
| 枚举选项     | 属性本身即枚举支持开关，取值仅 True/False | 已覆盖（test_type_is_bool、            |
|              |                                           | test_true_zero_ndims、test_false_*)    |
| 参数类型     | 返回类型必须为 Python bool                | 已覆盖（test_type_is_bool）            |
| 传参与不传参 | 无参属性，仅默认读取                      | 已覆盖（各用例均直接读属性）           |
| 等价类/边界  | ndims=0 边界（透传）与 ndims>0 边界（恒   | 已覆盖（test_true_zero_ndims、         |
|              | False）                                   | test_false_positive_ndims）            |
| 正常传参     | 与 base_dist.has_enumerate_support 逐例   | 已覆盖（test_true_zero_ndims、         |
|              | 一致                                      | test_false_when_base_lacks）           |
| 异常传参     | 无入参属性；ndims>0 下 enumerate_support  | 已覆盖（test_enumerate_support_raises_ |
|              | 调用的行为契约（NotImplementedError）     | for_positive_ndims）                   |
| 混合设备类型 | 纯 Python 布尔契约，不涉及设备张量        | 不适用                                 |

未覆盖项及原因：
- 无（上述 8 维中不适用项均已给出原因，其余均有具体用例与断言对应）。

注意：断言全部落在「返回类型 bool、与源码实现一致的 True/False 取值、
     ndims>0 下 enumerate_support 抛 NotImplementedError 的行为后果」三类
     可证伪契约上，取值断言使用 assertIs 与源码布尔字面量严格对照。
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
class TestIndependentHasEnumerateSupport(TestCase):
    """Test cases for torch.distributions.independent.Independent.has_enumerate_support."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_type_is_bool(self):
        """has_enumerate_support 的返回类型必须是 Python bool。"""
        base = torch.distributions.Categorical(
            torch.tensor([0.3, 0.7], device=self.device))
        ind = torch.distributions.Independent(base, 0)
        self.assertIsInstance(ind.has_enumerate_support, bool)

    def test_true_zero_ndims(self):
        """边界 ndims=0：与 base_dist.has_enumerate_support 严格一致（Categorical 为 True）。"""
        base = torch.distributions.Categorical(
            torch.tensor([0.3, 0.7], device=self.device))
        self.assertIs(base.has_enumerate_support, True)
        ind = torch.distributions.Independent(base, 0)
        self.assertIs(ind.has_enumerate_support, True)

    def test_false_positive_ndims(self):
        """边界 ndims>0：即使基分布可枚举，结果也恒为 False（源码 L90-L91）。"""
        base = torch.distributions.Categorical(
            torch.tensor([[0.3, 0.7]], device=self.device))
        self.assertEqual(base.batch_shape, torch.Size([1]))
        ind = torch.distributions.Independent(base, 1)
        self.assertIs(ind.has_enumerate_support, False)

    def test_false_when_base_lacks(self):
        """基分布不可枚举（Normal）时，ndims=0 也为 False。"""
        base = torch.distributions.Normal(
            torch.zeros(3, device=self.device), torch.ones(3, device=self.device))
        self.assertIs(base.has_enumerate_support, False)
        ind = torch.distributions.Independent(base, 0)
        self.assertIs(ind.has_enumerate_support, False)

    def test_enumerate_support_raises_for_positive_ndims(self):
        """行为后果：ndims>0 时调用 enumerate_support 必须抛 NotImplementedError。"""
        base = torch.distributions.Categorical(
            torch.tensor([[0.3, 0.7]], device=self.device))
        ind = torch.distributions.Independent(base, 1)
        with self.assertRaises(NotImplementedError):
            ind.enumerate_support()

    def test_enumerate_support_delegates_for_zero_ndims(self):
        """行为后果：ndims=0 时 enumerate_support 透传基分布结果。"""
        base = torch.distributions.Categorical(
            torch.tensor([0.3, 0.7], device=self.device))
        ind = torch.distributions.Independent(base, 0)
        torch.testing.assert_close(
            ind.enumerate_support(), base.enumerate_support())


if __name__ == "__main__":
    run_tests()
