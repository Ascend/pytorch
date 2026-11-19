# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.LazyLinear.initialize_parameters 接口功能正确性
API 名称：torch.nn.LazyLinear.initialize_parameters
API 签名：initialize_parameters(self, input) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 未初始化（UninitializedParameter）与已物化两种状态分别验证 | 已覆盖（infers_in_features/second_call_no_op） |
| 枚举选项 | 不涉及枚举参数 | 不适用（无枚举参数） |
| 参数类型 | input 为 2D 浮点 Tensor；in_features 按 input.shape[-1] 推断 | 已覆盖（test_initialize_parameters_infers_in_features） |
| 传参与不传参 | input 为必选位置参数；无默认值路径 | 已覆盖（test_initialize_parameters_infers_in_features） |
| 等价类/边界值 | 已物化但 in_features=0 时输入宽度不匹配抛 AssertionError、匹配则回填 | 已覆盖（mismatch_after_load_state_dict） |
| 正常传参场景 | 首次调用推断并物化参数；二次调用不重复初始化 | 已覆盖（infers_in_features/second_call_no_op） |
| 异常传参场景 | 已物化 + in_features=0 + 输入宽度不匹配 → AssertionError | 已覆盖（mismatch_after_load_state_dict） |
| 混合设备类型 | 设备构造+设备输入 → 参数落设备，前向与同权重 CPU Golden 对照 | 已覆盖（device_forward_matches_golden） |

未覆盖项及原因：
- 本文件同时看护别名路径 torch.nn.modules.linear.LazyLinear.initialize_parameters：
  torch.nn.LazyLinear 与 torch.nn.modules.linear.LazyLinear 为同一类对象，见 identity 断言用例。
- 3D 输入仅取 shape[-1]，未单独设例：行为与 2D 一致（仅读取最后一维），无独立契约面。

注意：CPU 侧验证种子确定性（两次同种子初始化精确一致）；设备侧浮点输出以同权重
     F.linear CPU 计算结果为 Golden，用 torch.testing.assert_close 对照；
     CPU 与设备随机数生成器不同，两侧参数值不做数值对照，设备侧按结构（形状/dtype/设备）断言。
"""
import math
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


def _is_lazy(param):
    """Return True when the parameter is still an UninitializedParameter."""
    return isinstance(param, torch.nn.parameter.UninitializedParameter)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestLazyLinearInitializeParameters(TestCase):
    """Comprehensive cases for torch.nn.LazyLinear.initialize_parameters (portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_dual_import_paths_same_object(self):
        """torch.nn.LazyLinear and torch.nn.modules.linear.LazyLinear are the same class."""
        self.assertIs(torch.nn.LazyLinear, torch.nn.modules.linear.LazyLinear)

    def test_initialize_parameters_infers_in_features(self):
        """First call infers in_features from input.shape[-1] and materializes params."""
        m = torch.nn.LazyLinear(5)
        self.assertTrue(m.has_uninitialized_params())
        self.assertTrue(_is_lazy(m.weight))
        x = torch.randn(2, 7)
        m.initialize_parameters(x)
        self.assertFalse(m.has_uninitialized_params())
        self.assertFalse(_is_lazy(m.weight))
        self.assertFalse(_is_lazy(m.bias))
        self.assertEqual(m.in_features, 7)
        self.assertEqual(tuple(m.weight.shape), (5, 7))
        self.assertEqual(tuple(m.bias.shape), (5,))
        self.assertIsInstance(m.weight, torch.nn.Parameter)

    def test_initialize_parameters_initializes_within_bounds(self):
        """Materialized values follow the Linear init distribution (bound 1/sqrt(fan_in))."""
        bound = 1.0 / math.sqrt(7)
        m = torch.nn.LazyLinear(5)
        m.initialize_parameters(torch.randn(2, 7))
        self.assertTrue(bool((m.weight.abs() <= bound).all()))
        self.assertTrue(bool((m.bias.abs() <= bound).all()))
        self.assertFalse(bool((m.weight == 0).all()))

    def test_initialize_parameters_deterministic_under_seed(self):
        """Two seeded lazy modules initialize to identical parameter values."""
        torch.manual_seed(21)
        first = torch.nn.LazyLinear(5)
        first.initialize_parameters(torch.randn(2, 7))
        torch.manual_seed(21)
        second = torch.nn.LazyLinear(5)
        second.initialize_parameters(torch.randn(2, 7))
        self.assertTrue(torch.equal(first.weight, second.weight))
        self.assertTrue(torch.equal(first.bias, second.bias))

    def test_initialize_parameters_second_call_no_op(self):
        """A second call on initialized params keeps values unchanged (no re-init)."""
        m = torch.nn.LazyLinear(5)
        m.initialize_parameters(torch.randn(2, 7))
        weight_snapshot = m.weight.detach().clone()
        bias_snapshot = m.bias.detach().clone()
        m.initialize_parameters(torch.randn(3, 7))
        self.assertTrue(torch.equal(m.weight.detach(), weight_snapshot))
        self.assertTrue(torch.equal(m.bias.detach(), bias_snapshot))

    def test_initialize_parameters_mismatch_assertion_after_load_state_dict(self):
        """Materialized-but-in_features=0 state: mismatched input width raises AssertionError.

        load_state_dict materializes the lazy parameters without setting
        in_features, so initialize_parameters enters the consistency branch:
        a mismatched input shape must raise AssertionError, a matching shape
        must set in_features without re-initializing.
        """
        m = torch.nn.LazyLinear(4)
        full = torch.nn.Linear(6, 4)
        m.load_state_dict(full.state_dict())
        self.assertFalse(m.has_uninitialized_params())
        self.assertEqual(m.in_features, 0)
        with self.assertRaisesRegex(AssertionError, "in_features inferred from input"):
            m.initialize_parameters(torch.randn(2, 5))
        m.initialize_parameters(torch.randn(2, 6))
        self.assertEqual(m.in_features, 6)
        self.assertTrue(torch.equal(m.weight.detach(), full.weight.detach()))
        self.assertTrue(torch.equal(m.bias.detach(), full.bias.detach()))

    def test_initialize_parameters_keeps_lazy_class_before_forward(self):
        """Direct initialize_parameters materializes params but keeps the lazy class.

        The class conversion via cls_to_become happens in the forward pre-hook
        (_infer_parameters), not in initialize_parameters itself.
        """
        m = torch.nn.LazyLinear(5)
        m.initialize_parameters(torch.randn(2, 7))
        self.assertIs(type(m), torch.nn.LazyLinear)
        self.assertFalse(m.has_uninitialized_params())

    def test_initialize_parameters_device_forward_matches_golden(self):
        """Device lazy module materializes params on the device; forward matches golden.

        Same-shape inputs drive the init on both sides; the parameter values use
        each side's own generator, so the golden is computed with F.linear from
        the device module's own weights instead of cross-side value comparison.
        """
        m = torch.nn.LazyLinear(5, device=self.device)
        x = torch.randn(2, 7, device=self.device)
        m.initialize_parameters(x)
        self.assertEqual(m.weight.device.type, self.device_name)
        self.assertEqual(m.bias.device.type, self.device_name)
        self.assertEqual(m.in_features, 7)
        out = m(x)
        self.assertEqual(tuple(out.shape), (2, 5))
        golden = torch.nn.functional.linear(
            x.detach().cpu(), m.weight.detach().cpu(), m.bias.detach().cpu())
        torch.testing.assert_close(out.detach().cpu(), golden)


if __name__ == "__main__":
    run_tests()
