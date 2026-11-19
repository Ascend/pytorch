# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.Linear.reset_parameters 接口功能正确性
API 名称：torch.nn.Linear.reset_parameters
API 签名：reset_parameters(self) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | bias=True/False 两种参数结构均重置成功 | 已覆盖（test_reset_parameters_bias_false） |
| 枚举选项 | bias 开关两分支；reset 后数值服从初始化分布 | 已覆盖（test_reset_parameters_bias_false/test_reset_parameters_bounds_on_device） |
| 参数类型 | weight/bias 形状、dtype、设备在重置前后保持不变 | 已覆盖（test_reset_parameters_preserves_param_objects） |
| 传参与不传参 | 无参方法，仅默认调用 | 已覆盖（test_reset_parameters_matches_fresh_construction） |
| 等价类/边界值 | 重置值精确等于同 RNG 状态全新 Linear 初始化值（上界 1/sqrt(fan_in)） | 已覆盖（matches_fresh_construction/bounds_on_device） |
| 正常传参场景 | 设备侧扰动后重置恢复初始化分布；重复调用参数对象引用不变 | 已覆盖（bounds_on_device/preserves_param_objects） |
| 异常传参场景 | 无参方法无校验路径；fan_in=0 非文档契约，行为随上游实现变化 | 未覆盖（in_features=0 仅 LazyLinear 内部依赖，非本 API 文档契约，未纳入） |
| 混合设备类型 | CPU 精确对照 + 设备侧（NPU/CUDA）重置与 RNG 续流精确对照 | 已覆盖（test_reset_parameters_device_continuation） |

未覆盖项及原因：
- in_features=0 的重置行为非文档契约（仅 LazyLinear 构造内部使用），不纳入稳定断言。
- 本文件同时看护别名路径 torch.nn.modules.linear.Linear.reset_parameters：
  torch.nn.Linear 与 torch.nn.modules.linear.Linear 为同一类对象，见 identity 断言用例。

注意：重置语义验证使用 RNG 续流精确对照——同种子下 A/B/C 三个模块，A、B 连续构造消耗
     两段随机数，C 重置后必须与 B 精确一致（torch.equal），该断言在实现改变初始化序列时失败；
     设备侧浮点值分布断言使用确定边界（1/sqrt(fan_in)），不做 CPU/NPU 数值对照
     （两侧随机数生成器不同，数值不可对照，按分布边界与 RNG 续流验证）。
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
class TestLinearResetParameters(TestCase):
    """Comprehensive cases for torch.nn.Linear.reset_parameters (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_dual_import_paths_same_object(self):
        """torch.nn.Linear and torch.nn.modules.linear.Linear are the same class."""
        self.assertIs(torch.nn.Linear, torch.nn.modules.linear.Linear)

    def test_reset_parameters_matches_fresh_construction(self):
        """Reset reproduces exactly what a fresh Linear draws next (RNG continuation).

        Sequence: seed; A draws D1. B (no reseed) draws D2. seed; C draws D1.
        C.reset_parameters() must draw the same post-D1 stream as B, so C == B
        exactly. This fails if the reset skips or reorders any init draw.
        """
        torch.manual_seed(99)
        first_draw = torch.nn.Linear(4, 3)  # A: consumes D1
        reference = torch.nn.Linear(4, 3)  # B: consumes D2
        torch.manual_seed(99)
        target = torch.nn.Linear(4, 3)  # C: consumes D1 again
        self.assertTrue(torch.equal(target.weight, first_draw.weight))
        self.assertFalse(torch.equal(target.weight, reference.weight))
        target.reset_parameters()
        self.assertTrue(torch.equal(target.weight, reference.weight))
        self.assertTrue(torch.equal(target.bias, reference.bias))

    def test_reset_parameters_deterministic_under_seed(self):
        """Two fresh constructions under the same seed produce identical params."""
        torch.manual_seed(42)
        first = torch.nn.Linear(5, 4)
        torch.manual_seed(42)
        second = torch.nn.Linear(5, 4)
        self.assertTrue(torch.equal(first.weight, second.weight))
        self.assertTrue(torch.equal(first.bias, second.bias))

    def test_reset_parameters_bounds_on_device(self):
        """After perturbation, reset restores values within the init bounds on device."""
        m = torch.nn.Linear(4, 3, device=self.device)
        with torch.no_grad():
            m.weight.fill_(999.0)
            m.bias.fill_(999.0)
        m.reset_parameters()
        bound = 1.0 / math.sqrt(4)
        self.assertTrue(bool((m.weight.abs() <= bound).all()))
        self.assertTrue(bool((m.bias.abs() <= bound).all()))
        self.assertTrue(bool((m.weight != 999.0).all()))
        self.assertTrue(bool((m.bias != 999.0).all()))

    def test_reset_parameters_preserves_param_objects(self):
        """Reset keeps the same Parameter objects, shapes and dtypes."""
        m = torch.nn.Linear(4, 3)
        weight_ref = m.weight
        bias_ref = m.bias
        m.reset_parameters()
        self.assertIs(m.weight, weight_ref)
        self.assertIs(m.bias, bias_ref)
        self.assertEqual(tuple(m.weight.shape), (3, 4))
        self.assertEqual(tuple(m.bias.shape), (3,))
        self.assertEqual(m.weight.dtype, torch.float32)

    def test_reset_parameters_bias_false(self):
        """A bias-free Linear resets its weight and keeps no bias attribute."""
        m = torch.nn.Linear(4, 3, bias=False)
        self.assertIsNone(m.bias)
        with torch.no_grad():
            m.weight.fill_(999.0)
        m.reset_parameters()
        bound = 1.0 / math.sqrt(4)
        self.assertTrue(bool((m.weight.abs() <= bound).all()))
        self.assertTrue(bool((m.weight != 999.0).all()))

    def test_reset_parameters_device_continuation(self):
        """RNG continuation equality also holds for device-allocated parameters."""
        torch.manual_seed(7)
        torch.nn.Linear(4, 3, device=self.device)  # A: consumes device draws D1
        reference = torch.nn.Linear(4, 3, device=self.device)  # B: D2
        torch.manual_seed(7)
        target = torch.nn.Linear(4, 3, device=self.device)  # C: D1 again
        target.reset_parameters()
        self.assertTrue(torch.equal(target.weight, reference.weight))
        self.assertTrue(torch.equal(target.bias, reference.bias))
        self.assertEqual(target.weight.device.type, self.device_name)

    def test_reset_parameters_forward_after_reset(self):
        """Forward output after reset matches the CPU golden computed from the
        same weights (identical values, compared with torch.testing.assert_close)."""
        m = torch.nn.Linear(4, 3, device=self.device)
        m.reset_parameters()
        x = torch.randn(2, 4, device=self.device)
        out = m(x)
        golden = torch.nn.functional.linear(x.detach().cpu(), m.weight.detach().cpu(),
                                            m.bias.detach().cpu())
        torch.testing.assert_close(out.detach().cpu(), golden)


if __name__ == "__main__":
    run_tests()
