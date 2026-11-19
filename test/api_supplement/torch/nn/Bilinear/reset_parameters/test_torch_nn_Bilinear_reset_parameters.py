# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.Bilinear.reset_parameters 接口功能正确性
API 名称：torch.nn.Bilinear.reset_parameters
API 签名：reset_parameters(self) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | bias=True/False 两种参数结构均重置成功 | 已覆盖（test_reset_parameters_bias_false） |
| 枚举选项 | bias 开关两分支；in1_features<=0 触发 ValueError | 已覆盖（test_reset_parameters_invalid_in1_features） |
| 参数类型 | weight/bias 形状、dtype 在重置前后保持不变 | 已覆盖（test_reset_parameters_preserves_param_objects） |
| 传参与不传参 | 无参方法，仅默认调用 | 已覆盖（test_reset_parameters_matches_fresh_construction） |
| 等价类/边界值 | 重置值精确等于同 RNG 状态全新 Bilinear 初始化值；上界 1/sqrt(in1) | 已覆盖（matches_fresh_construction/bounds_on_device） |
| 正常传参场景 | 设备侧扰动后重置恢复初始化分布，参数对象引用不变 | 已覆盖（bounds_on_device/preserves_param_objects） |
| 异常传参场景 | in1_features 置 0 或负数后重置抛 ValueError（源码契约） | 已覆盖（test_reset_parameters_invalid_in1_features） |
| 混合设备类型 | CPU 精确对照 + 设备侧重置与 RNG 续流精确对照 | 已覆盖（test_reset_parameters_device_continuation） |

未覆盖项及原因：
- 本文件同时看护别名路径 torch.nn.modules.linear.Bilinear.reset_parameters：
  torch.nn.Bilinear 与 torch.nn.modules.linear.Bilinear 为同一类对象，见 identity 断言用例。

注意：重置语义验证使用 RNG 续流精确对照（seed; A 构造消耗 D1；B 续流构造消耗 D2；
     seed 后 C 构造回到 D1，C.reset_parameters() 后必须与 B 精确一致）；
     设备侧断言使用确定分布边界 1/sqrt(in1_features)，不做 CPU/NPU 数值对照
     （两侧随机数生成器不同）。
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


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestBilinearResetParameters(TestCase):
    """Comprehensive cases for torch.nn.Bilinear.reset_parameters (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_dual_import_paths_same_object(self):
        """torch.nn.Bilinear and torch.nn.modules.linear.Bilinear are the same class."""
        self.assertIs(torch.nn.Bilinear, torch.nn.modules.linear.Bilinear)

    def test_reset_parameters_matches_fresh_construction(self):
        """Reset reproduces exactly what a fresh Bilinear draws next (RNG continuation)."""
        torch.manual_seed(11)
        first_draw = torch.nn.Bilinear(5, 6, 3)  # A: consumes D1
        reference = torch.nn.Bilinear(5, 6, 3)  # B: consumes D2
        torch.manual_seed(11)
        target = torch.nn.Bilinear(5, 6, 3)  # C: consumes D1 again
        self.assertTrue(torch.equal(target.weight, first_draw.weight))
        self.assertFalse(torch.equal(target.weight, reference.weight))
        target.reset_parameters()
        self.assertTrue(torch.equal(target.weight, reference.weight))
        self.assertTrue(torch.equal(target.bias, reference.bias))

    def test_reset_parameters_bounds_on_device(self):
        """After perturbation, reset restores values within 1/sqrt(in1) on device."""
        m = torch.nn.Bilinear(5, 6, 3, device=self.device)
        with torch.no_grad():
            m.weight.fill_(999.0)
            m.bias.fill_(999.0)
        m.reset_parameters()
        bound = 1.0 / math.sqrt(5)
        self.assertTrue(bool((m.weight.abs() <= bound).all()))
        self.assertTrue(bool((m.bias.abs() <= bound).all()))
        self.assertTrue(bool((m.weight != 999.0).all()))
        self.assertTrue(bool((m.bias != 999.0).all()))

    def test_reset_parameters_preserves_param_objects(self):
        """Reset keeps the same Parameter objects, shapes and dtypes."""
        m = torch.nn.Bilinear(5, 6, 3)
        weight_ref = m.weight
        bias_ref = m.bias
        m.reset_parameters()
        self.assertIs(m.weight, weight_ref)
        self.assertIs(m.bias, bias_ref)
        self.assertEqual(tuple(m.weight.shape), (3, 5, 6))
        self.assertEqual(tuple(m.bias.shape), (3,))
        self.assertEqual(m.weight.dtype, torch.float32)

    def test_reset_parameters_bias_false(self):
        """A bias-free Bilinear resets its weight and keeps no bias attribute."""
        m = torch.nn.Bilinear(5, 6, 3, bias=False)
        self.assertIsNone(m.bias)
        with torch.no_grad():
            m.weight.fill_(999.0)
        m.reset_parameters()
        bound = 1.0 / math.sqrt(5)
        self.assertTrue(bool((m.weight.abs() <= bound).all()))
        self.assertTrue(bool((m.weight != 999.0).all()))

    def test_reset_parameters_invalid_in1_features(self):
        """in1_features <= 0 makes reset_parameters raise ValueError (source contract)."""
        m_zero = torch.nn.Bilinear(5, 6, 3)
        m_zero.in1_features = 0
        with self.assertRaisesRegex(ValueError, "in1_features must be > 0"):
            m_zero.reset_parameters()
        m_negative = torch.nn.Bilinear(5, 6, 3)
        m_negative.in1_features = -1
        with self.assertRaisesRegex(ValueError, "in1_features must be > 0"):
            m_negative.reset_parameters()

    def test_reset_parameters_device_continuation(self):
        """RNG continuation equality also holds for device-allocated parameters."""
        torch.manual_seed(13)
        torch.nn.Bilinear(5, 6, 3, device=self.device)  # A: D1
        reference = torch.nn.Bilinear(5, 6, 3, device=self.device)  # B: D2
        torch.manual_seed(13)
        target = torch.nn.Bilinear(5, 6, 3, device=self.device)  # C: D1 again
        target.reset_parameters()
        self.assertTrue(torch.equal(target.weight, reference.weight))
        self.assertTrue(torch.equal(target.bias, reference.bias))
        self.assertEqual(target.weight.device.type, self.device_name)

    def test_reset_parameters_forward_after_reset(self):
        """Forward output after reset matches the CPU golden from the same weights."""
        m = torch.nn.Bilinear(5, 6, 3, device=self.device)
        m.reset_parameters()
        x1 = torch.randn(2, 5, device=self.device)
        x2 = torch.randn(2, 6, device=self.device)
        out = m(x1, x2)
        golden = torch.nn.functional.bilinear(
            x1.detach().cpu(), x2.detach().cpu(),
            m.weight.detach().cpu(), m.bias.detach().cpu())
        torch.testing.assert_close(out.detach().cpu(), golden)


if __name__ == "__main__":
    run_tests()
