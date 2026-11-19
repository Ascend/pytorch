# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.modules.linear.LazyLinear.cls_to_become 接口功能正确性
API 名称：torch.nn.modules.linear.LazyLinear.cls_to_become
API 签名：cls_to_become: type[torch.nn.Linear]（LazyModuleMixin 类属性，LazyLinear 覆写为 nn.Linear）

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | cls_to_become 非 None（LazyModuleMixin 默认 None，LazyLinear 必须覆写） | 已覆盖（attribute_value/mixin_default_is_none） |
| 枚举选项 | 不涉及枚举参数 | 不适用（类属性无枚举值） |
| 参数类型 | 属性值为类对象且恰为 torch.nn.Linear（identity） | 已覆盖（test_cls_to_become_attribute_value） |
| 传参与不传参 | 类属性访问；实例在 forward 物化后按该属性完成类型转换 | 已覆盖（test_instance_converts_after_dry_run） |
| 等价类/边界值 | 目标类自身不再携带 cls_to_become；转换后实例失去该属性 | 已覆盖（target_class_has_no_attr/instance_converts） |
| 正常传参场景 | 设备侧 dry run 一次前向即触发物化与类型转换，参数落设备 | 已覆盖（test_instance_converts_after_dry_run） |
| 异常传参场景 | 不适用（属性访问无异常契约） | 不适用（无异常路径） |
| 混合设备类型 | 设备构造 + 设备前向，转换后 in_features/weight 形状与前向 Golden 对照 | 已覆盖（test_instance_converts_after_dry_run） |

未覆盖项及原因：
- 异常传参场景不适用：cls_to_become 是类属性，无可调用契约与异常路径。
- 未覆盖多次 dry run：首次 forward 即移除物化 hook 并完成类型转换，重复触发无独立契约面。

注意：cls_to_become 的可观测语义是「lazy 模块首次前向物化后 __class__ 变为目标类」，
     用例以 type(instance) identity 断言验证转换确实发生；前向浮点输出以同权重
     F.linear CPU 结果为 Golden，用 torch.testing.assert_close 对照。
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
class TestLazyLinearClsToBecome(TestCase):
    """Comprehensive cases for LazyLinear.cls_to_become (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_cls_to_become_attribute_value(self):
        """LazyLinear.cls_to_become is exactly torch.nn.Linear."""
        self.assertIs(torch.nn.modules.linear.LazyLinear.cls_to_become,
                      torch.nn.Linear)
        self.assertIs(torch.nn.LazyLinear.cls_to_become, torch.nn.Linear)

    def test_lazy_module_mixin_default_is_none(self):
        """LazyModuleMixin defaults cls_to_become to None; LazyLinear overrides it."""
        self.assertIsNone(torch.nn.modules.lazy.LazyModuleMixin.cls_to_become)
        self.assertIsNotNone(torch.nn.modules.linear.LazyLinear.cls_to_become)

    def test_target_class_has_no_cls_to_become(self):
        """The target Linear class itself does not carry cls_to_become."""
        self.assertFalse(hasattr(torch.nn.Linear, "cls_to_become"))

    def test_instance_converts_after_dry_run(self):
        """One dry-run forward materializes params and swaps __class__ to Linear."""
        m = torch.nn.modules.linear.LazyLinear(5, device=self.device)
        self.assertIs(type(m), torch.nn.modules.linear.LazyLinear)
        self.assertIsInstance(m.weight, torch.nn.parameter.UninitializedParameter)
        self.assertIsInstance(m.bias, torch.nn.parameter.UninitializedParameter)
        x = torch.randn(2, 7, device=self.device)
        out = m(x)
        self.assertIs(type(m), torch.nn.Linear)
        self.assertFalse(hasattr(m, "cls_to_become"))
        self.assertNotIsInstance(m.weight, torch.nn.parameter.UninitializedParameter)
        self.assertNotIsInstance(m.bias, torch.nn.parameter.UninitializedParameter)
        self.assertEqual(m.in_features, 7)
        self.assertEqual(tuple(m.weight.shape), (5, 7))
        self.assertEqual(m.weight.device.type, self.device_name)
        self.assertEqual(tuple(out.shape), (2, 5))
        golden = torch.nn.functional.linear(
            x.detach().cpu(), m.weight.detach().cpu(), m.bias.detach().cpu())
        torch.testing.assert_close(out.detach().cpu(), golden)

    def test_conversion_requires_forward_not_construction(self):
        """Constructing with an input-shaped tensor is impossible; only forward infers."""
        m = torch.nn.modules.linear.LazyLinear(5)
        self.assertIs(type(m), torch.nn.modules.linear.LazyLinear)
        self.assertEqual(m.in_features, 0)
        self.assertTrue(m.has_uninitialized_params())


if __name__ == "__main__":
    run_tests()
