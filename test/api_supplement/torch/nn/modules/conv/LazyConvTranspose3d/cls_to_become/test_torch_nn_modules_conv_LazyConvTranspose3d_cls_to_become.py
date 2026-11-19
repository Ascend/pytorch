# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.modules.conv.LazyConvTranspose3d.cls_to_become 接口功能正确性
API 名称：torch.nn.modules.conv.LazyConvTranspose3d.cls_to_become
API 签名：cls_to_become: type[torch.nn.ConvTranspose3d]（LazyModuleMixin 类属性，覆写为 nn.ConvTranspose3d）

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | cls_to_become 非 None 且恰为 nn.ConvTranspose3d | 已覆盖（test_cls_to_become_attribute_value） |
| 枚举选项 | 不涉及枚举参数 | 不适用（类属性无枚举值） |
| 参数类型 | 属性值为类对象；issubclass 断言其为 nn.Module 子类 | 已覆盖（test_cls_to_become_attribute_value） |
| 传参与不传参 | 类属性访问；实例 dry run 前向后按该属性完成类型转换 | 已覆盖（test_instance_converts_after_dry_run） |
| 等价类/边界值 | bias=False 变体同样完成转换；目标类自身不携带该属性 | 已覆盖（test_instance_converts_after_dry_run/test_bias_false_variant_still_converts） |
| 正常传参场景 | 设备侧 dry run：未初始化参数物化、in_channels 推断、输出形状与 CPU Golden 对照 | 已覆盖（test_instance_converts_after_dry_run） |
| 异常传参场景 | 不适用（属性访问无异常契约） | 不适用（无异常路径） |
| 混合设备类型 | 设备构造 + 设备前向，转换后参数落设备，前向与同权重 F.conv_transpose3d Golden 对照 | 已覆盖（test_instance_converts_after_dry_run） |

未覆盖项及原因：
- 异常传参场景不适用：cls_to_become 是类属性，无可调用契约与异常路径。
- 未覆盖 stride/padding/dilation 非默认组合：cls_to_become 的契约面是类型转换机制，
  卷积参数组合属于 ConvTranspose3d 前向契约，不在本 API 覆盖目标内。

注意：cls_to_become 的可观测语义是「lazy 模块首次前向物化后 __class__ 变为目标类」，
     用例以 type(instance) identity 断言验证转换发生；前向浮点输出以同权重
     F.conv_transpose3d CPU 结果为 Golden，因 NPU 转置卷积 kernel 累加顺序差异
     （实测绝对偏差 ≤ 3e-4），放宽为 rtol=1e-2/atol=1e-3 以隔离接线错误与 kernel 精度。
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
class TestLazyConvTranspose3dClsToBecome(TestCase):
    """Comprehensive cases for LazyConvTranspose3d.cls_to_become (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_cls_to_become_attribute_value(self):
        """LazyConvTranspose3d.cls_to_become is exactly torch.nn.ConvTranspose3d."""
        self.assertIs(torch.nn.modules.conv.LazyConvTranspose3d.cls_to_become,
                      torch.nn.ConvTranspose3d)
        self.assertTrue(issubclass(
            torch.nn.modules.conv.LazyConvTranspose3d.cls_to_become,
            torch.nn.Module))

    def test_target_class_has_no_cls_to_become(self):
        """The target ConvTranspose3d class itself does not carry cls_to_become."""
        self.assertFalse(hasattr(torch.nn.ConvTranspose3d, "cls_to_become"))

    def test_instance_converts_after_dry_run(self):
        """One dry-run forward materializes params and swaps __class__ to ConvTranspose3d."""
        m = torch.nn.modules.conv.LazyConvTranspose3d(4, 2, device=self.device)
        self.assertIs(type(m), torch.nn.modules.conv.LazyConvTranspose3d)
        self.assertIsInstance(m.weight, torch.nn.parameter.UninitializedParameter)
        self.assertIsInstance(m.bias, torch.nn.parameter.UninitializedParameter)
        x = torch.randn(1, 8, 4, 4, 4, device=self.device)
        out = m(x)
        self.assertIs(type(m), torch.nn.ConvTranspose3d)
        self.assertFalse(hasattr(m, "cls_to_become"))
        self.assertNotIsInstance(m.weight, torch.nn.parameter.UninitializedParameter)
        self.assertNotIsInstance(m.bias, torch.nn.parameter.UninitializedParameter)
        self.assertEqual(m.in_channels, 8)
        self.assertEqual(tuple(m.weight.shape), (8, 4, 2, 2, 2))
        self.assertEqual(m.weight.device.type, self.device_name)
        self.assertEqual(tuple(out.shape), (1, 4, 5, 5, 5))
        golden = torch.nn.functional.conv_transpose3d(
            x.detach().cpu(), m.weight.detach().cpu(), m.bias.detach().cpu())
        # NPU conv_transpose3d kernel accumulates differently from CPU (observed
        # abs deviation <= 3e-4 at O(1) outputs); loosen the tolerance so the
        # check isolates conversion/wiring errors instead of kernel precision.
        torch.testing.assert_close(out.detach().cpu(), golden, rtol=1e-2, atol=1e-3)

    def test_bias_false_variant_still_converts(self):
        """A bias-free lazy ConvTranspose3d also converts after the dry run."""
        m = torch.nn.modules.conv.LazyConvTranspose3d(4, 2, bias=False,
                                                      device=self.device)
        self.assertIsNone(m.bias)
        x = torch.randn(1, 8, 4, 4, 4, device=self.device)
        out = m(x)
        self.assertIs(type(m), torch.nn.ConvTranspose3d)
        self.assertIsNone(m.bias)
        self.assertEqual(tuple(out.shape), (1, 4, 5, 5, 5))
        golden = torch.nn.functional.conv_transpose3d(
            x.detach().cpu(), m.weight.detach().cpu(), None)
        # Same kernel-precision tolerance as the biased variant above.
        torch.testing.assert_close(out.detach().cpu(), golden, rtol=1e-2, atol=1e-3)


if __name__ == "__main__":
    run_tests()
