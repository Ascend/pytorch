# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.nn.qat.dynamic.Linear 接口功能正确性
API 名称：torch.ao.nn.qat.dynamic.Linear
API 签名：Linear(in_features, out_features, bias=True, qconfig=None, device=None, dtype=None)

覆盖维度表：
| 覆盖维度      | 说明                                        | 覆盖情况                                           |
|---------------|---------------------------------------------|----------------------------------------------------|
| 空/非空       | qconfig 取 None（空）与合法 memoryless      | 已覆盖：test_init_none_qconfig_raises /            |
|               | qconfig（非空）                             | test_init_returns_qat_linear                       |
| 枚举选项      | 无枚举参数；bias 取 True/False 两态         | 已覆盖：test_init_default_bias /                   |
|               |                                             | test_init_bias_false                               |
| 参数类型      | in_features/out_features 为 int，bias 为    | 已覆盖：test_init_copies_feature_dims /            |
|               | bool，qconfig 为 QConfig                    | test_weight_fake_quant_is_fake_quantize            |
| 传参与不传参  | 省略 bias 用默认 True；显式传 bias=False    | 已覆盖：test_init_default_bias /                   |
|               |                                             | test_init_bias_false                               |
| 等价类/边界值 | 典型 (4,5) 组合与 no-bias 变体              | 已覆盖：test_init_copies_feature_dims              |
| 正常传参场景  | 以 memoryless qconfig 构造并前向 + 反传     | 已覆盖：test_forward_output_shape_and_grad         |
| 异常传参场景  | 非 memoryless qconfig 抛 ValueError；       | 已覆盖：test_init_non_memoryless_qconfig_raises /  |
|               | qconfig=None 抛 AssertionError              | test_init_none_qconfig_raises                      |
| 混合设备类型  | 模块构造/前向接口，不涉及多 Tensor          | 不适用                                             |
|               | 异构设备输入                                |                                                    |

未覆盖项及原因：
- 混合设备类型：动态 QAT Linear 仅由 CPU 量化后端提供 FakeQuantize 语义，无 NPU/CUDA 执行路径，
  无法构造异构设备输入。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注：动态 QAT Linear 只在 CPU 量化后端上有 fake-quant 语义，无 NPU/CUDA 数值对照路径，
     故不做 CPU/NPU Golden 对照，仅验证构造校验与 fake-quant 前向/反传语义。
"""
import os
import unittest

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Probe the device only when the supplement suite actually runs. On the default
# CI collection path the class-level skipUnless fires first, so no torch_npu
# import or device initialization happens here.
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
        DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

import torch.ao.nn.qat
import torch.ao.quantization
from torch.ao.nn.qat.dynamic import Linear as DynamicQATLinear

_MEMORYLESS_QCONFIG = torch.ao.quantization.default_dynamic_qat_qconfig
_NON_MEMORYLESS_QCONFIG = torch.ao.quantization.default_qat_qconfig


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestTorchAoNnQatDynamicLinearUt(TestCase):
    """Functional tests for torch.ao.nn.qat.dynamic.Linear (CPU-only QAT semantics)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_init_returns_qat_linear(self):
        """Construction with a memoryless qconfig yields a QAT Linear with the given shape."""
        module = DynamicQATLinear(4, 5, qconfig=_MEMORYLESS_QCONFIG)
        self.assertIsInstance(module, torch.ao.nn.qat.Linear)
        self.assertEqual(tuple(module.weight.shape), (5, 4))

    def test_init_copies_feature_dims(self):
        """in_features/out_features are stored on the module and drive the weight layout."""
        module = DynamicQATLinear(7, 3, qconfig=_MEMORYLESS_QCONFIG)
        self.assertEqual(module.in_features, 7)
        self.assertEqual(module.out_features, 3)
        self.assertEqual(tuple(module.weight.shape), (3, 7))

    def test_init_default_bias(self):
        """Omitting bias uses the documented default True so a bias parameter exists."""
        module = DynamicQATLinear(4, 5, qconfig=_MEMORYLESS_QCONFIG)
        self.assertIsNotNone(module.bias)
        self.assertEqual(tuple(module.bias.shape), (5,))

    def test_init_bias_false(self):
        """Explicit bias=False removes the bias parameter."""
        module = DynamicQATLinear(4, 5, bias=False, qconfig=_MEMORYLESS_QCONFIG)
        self.assertIsNone(module.bias)

    def test_weight_fake_quant_is_fake_quantize(self):
        """The weight fake-quant module is built from the qconfig weight entry."""
        module = DynamicQATLinear(4, 5, qconfig=_MEMORYLESS_QCONFIG)
        self.assertIsInstance(module.weight_fake_quant, torch.ao.quantization.FakeQuantize)
        # The module must actually hold the user supplied qconfig object.
        self.assertIs(module.qconfig, _MEMORYLESS_QCONFIG)

    def test_forward_output_shape_and_grad(self):
        """Forward returns the Linear output shape and fake-quant keeps gradients flowing."""
        module = DynamicQATLinear(4, 5, qconfig=_MEMORYLESS_QCONFIG)
        x = torch.randn(2, 4)
        out = module(x)
        self.assertEqual(tuple(out.shape), (2, 5))
        self.assertEqual(out.dtype, torch.float32)
        out.sum().backward()
        # A missing/incorrect fake-quant wiring would break the gradient path.
        self.assertIsNotNone(module.weight.grad)
        self.assertIsNotNone(module.bias.grad)

    def test_init_non_memoryless_qconfig_raises(self):
        """A non-memoryless activation observer violates the dynamic QAT contract."""
        with self.assertRaisesRegex(ValueError, "memoryless observer"):
            DynamicQATLinear(4, 5, qconfig=_NON_MEMORYLESS_QCONFIG)

    def test_init_none_qconfig_raises(self):
        """Omitting the qconfig is rejected by the QAT module contract."""
        with self.assertRaisesRegex(AssertionError, "qconfig must be provided for QAT module"):
            DynamicQATLinear(4, 5, qconfig=None)


if __name__ == "__main__":
    run_tests()
