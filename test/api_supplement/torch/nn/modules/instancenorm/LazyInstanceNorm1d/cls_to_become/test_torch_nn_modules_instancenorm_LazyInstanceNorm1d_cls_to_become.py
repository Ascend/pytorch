# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.modules.instancenorm.LazyInstanceNorm1d.cls_to_become 接口功能正确性
API 名称：torch.nn.modules.instancenorm.LazyInstanceNorm1d.cls_to_become
API 签名：cls_to_become: type[torch.nn.InstanceNorm1d]（LazyModuleMixin 类属性，覆写为 nn.InstanceNorm1d）

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | cls_to_become 非 None 且恰为 InstanceNorm1d；dry run 前参数/缓冲未初始化 | 已覆盖（attribute_value/instance_converts） |
| 枚举选项 | 不涉及枚举参数 | 不适用（类属性无枚举值） |
| 参数类型 | 属性值为类对象；issubclass 断言其为 nn.Module 子类 | 已覆盖（test_cls_to_become_attribute_value） |
| 传参与不传参 | 类属性访问；实例 dry run 前向后按该属性完成类型转换 | 已覆盖（test_instance_converts_after_dry_run） |
| 等价类/边界值 | num_features 由输入通道数推断；目标类自身不携带该属性 | 已覆盖（instance_converts/target_class_has_no_attr） |
| 正常传参场景 | 设备侧 dry run：参数/统计缓冲物化、类型转换、转换后前向与 F.instance_norm CPU Golden 对照 | 已覆盖（test_instance_converts_after_dry_run） |
| 异常传参场景 | 不适用（属性访问无异常契约） | 不适用（无异常路径） |
| 混合设备类型 | 设备构造 + 设备前向，转换后参数与统计缓冲落设备 | 已覆盖（test_instance_converts_after_dry_run） |

未覆盖项及原因：
- 异常传参场景不适用：cls_to_become 是类属性，无可调用契约与异常路径。
- affine=False/track_running_stats=False 变体未单独设例：转换机制由 LazyModuleMixin
  统一实现，与本属性契约无关，默认配置已覆盖转换语义。

注意：cls_to_become 的可观测语义是「lazy 模块首次前向物化后 __class__ 变为目标类」，
     用例以 type(instance) identity 断言验证转换发生；训练态 instance norm 输出由
     batch 统计决定，转换后前向与同权重 F.instance_norm CPU 结果为 Golden，
     用 torch.testing.assert_close 对照。
"""
import os
import unittest

import torch

RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"
SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"

DEVICE_NAME = "cpu"
if RUN_FLAG:
    DEVICE_NAME = None
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = "npu"
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        if not torch.cuda.is_available():
            raise RuntimeError("No available NPU or GPU device.")
        DEVICE_NAME = "cuda"

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestLazyInstanceNorm1dClsToBecome(TestCase):
    """Comprehensive cases for LazyInstanceNorm1d.cls_to_become (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_cls_to_become_attribute_value(self):
        """LazyInstanceNorm1d.cls_to_become is exactly torch.nn.InstanceNorm1d."""
        self.assertIs(torch.nn.modules.instancenorm.LazyInstanceNorm1d.cls_to_become,
                      torch.nn.InstanceNorm1d)
        self.assertTrue(issubclass(
            torch.nn.modules.instancenorm.LazyInstanceNorm1d.cls_to_become,
            torch.nn.Module))

    def test_target_class_has_no_cls_to_become(self):
        """The target InstanceNorm1d class itself does not carry cls_to_become."""
        self.assertFalse(hasattr(torch.nn.InstanceNorm1d, "cls_to_become"))

    def test_instance_converts_after_dry_run(self):
        """One dry-run forward materializes params and swaps __class__ to InstanceNorm1d."""
        m = torch.nn.modules.instancenorm.LazyInstanceNorm1d(
            affine=True, track_running_stats=True, device=self.device)
        self.assertIs(type(m), torch.nn.modules.instancenorm.LazyInstanceNorm1d)
        self.assertIsInstance(m.weight, torch.nn.parameter.UninitializedParameter)
        self.assertIsInstance(m.bias, torch.nn.parameter.UninitializedParameter)
        self.assertIsInstance(m.running_mean, torch.nn.parameter.UninitializedBuffer)
        self.assertIsInstance(m.running_var, torch.nn.parameter.UninitializedBuffer)
        x = torch.randn(2, 6, 10, device=self.device)
        out = m(x)
        self.assertIs(type(m), torch.nn.InstanceNorm1d)
        self.assertFalse(hasattr(m, "cls_to_become"))
        self.assertNotIsInstance(m.weight, torch.nn.parameter.UninitializedParameter)
        self.assertNotIsInstance(m.bias, torch.nn.parameter.UninitializedParameter)
        self.assertNotIsInstance(m.running_mean,
                                 torch.nn.parameter.UninitializedBuffer)
        self.assertNotIsInstance(m.running_var,
                                 torch.nn.parameter.UninitializedBuffer)
        self.assertEqual(m.num_features, 6)
        self.assertEqual(tuple(m.weight.shape), (6,))
        self.assertEqual(tuple(m.running_mean.shape), (6,))
        self.assertEqual(m.weight.device.type, self.device_name)
        self.assertEqual(m.running_mean.device.type, self.device_name)
        self.assertEqual(tuple(out.shape), (2, 6, 10))
        out_again = m(x)
        torch.testing.assert_close(out_again.detach().cpu(), out.detach().cpu())
        golden = torch.nn.functional.instance_norm(
            x.detach().cpu(), weight=m.weight.detach().cpu(),
            bias=m.bias.detach().cpu())
        torch.testing.assert_close(out.detach().cpu(), golden)


if __name__ == "__main__":
    run_tests()
