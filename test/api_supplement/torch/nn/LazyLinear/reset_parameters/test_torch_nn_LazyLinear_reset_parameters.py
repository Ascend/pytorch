# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.LazyLinear.reset_parameters 接口功能正确性
API 名称：torch.nn.LazyLinear.reset_parameters
API 签名：reset_parameters(self) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 未初始化（UninitializedParameter）调用为无操作；已物化调用执行重置 | 已覆盖（noop_when_uninitialized/after_initialization） |
| 枚举选项 | 不涉及枚举参数 | 不适用（无枚举参数） |
| 参数类型 | 重置后 weight/bias 形状、dtype、设备保持不变 | 已覆盖（test_reset_parameters_preserves_objects_on_device） |
| 传参与不传参 | 无参方法，仅默认调用 | 已覆盖（test_reset_parameters_noop_when_uninitialized） |
| 等价类/边界值 | 重置值精确等于同 RNG 状态全新 Linear 初始化值；上界 1/sqrt(in_features) | 已覆盖（matches_fresh_linear_init/after_initialization） |
| 正常传参场景 | 物化后重置恢复 Linear 初始化分布；完整 forward 转为 Linear 后重置仍可用 | 已覆盖（test_reset_parameters_after_full_forward） |
| 异常传参场景 | 无参方法无校验路径；未初始化时静默跳过（源码守卫契约） | 已覆盖（test_reset_parameters_noop_when_uninitialized） |
| 混合设备类型 | 设备侧物化后扰动重置，分布边界与设备保持断言 + 前向 Golden 对照 | 已覆盖（test_reset_parameters_on_device） |

未覆盖项及原因：
- 本文件同时看护别名路径 torch.nn.modules.linear.LazyLinear.reset_parameters：
  torch.nn.LazyLinear 与 torch.nn.modules.linear.LazyLinear 为同一类对象，见 identity 断言用例。

注意：LazyLinear.reset_parameters 的契约是「未初始化时不做任何事、已初始化时委托
     Linear.reset_parameters」。CPU 精确对照使用 RNG 续流：物化消耗 D1 后，全新
     nn.Linear(in,out) 续流消耗 D2，重置后的模块必须与该 Linear 精确一致；
     设备侧随机数生成器与 CPU 不同，按分布边界、对象同一性与前向 Golden 验证。
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
class TestLazyLinearResetParameters(TestCase):
    """Comprehensive cases for torch.nn.LazyLinear.reset_parameters (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_dual_import_paths_same_object(self):
        """torch.nn.LazyLinear and torch.nn.modules.linear.LazyLinear are the same class."""
        self.assertIs(torch.nn.LazyLinear, torch.nn.modules.linear.LazyLinear)

    def test_reset_parameters_noop_when_uninitialized(self):
        """Calling reset on an uninitialized lazy module is a guarded no-op."""
        m = torch.nn.LazyLinear(5)
        m.reset_parameters()
        self.assertTrue(m.has_uninitialized_params())
        self.assertTrue(_is_lazy(m.weight))
        self.assertTrue(_is_lazy(m.bias))

    def test_reset_parameters_matches_fresh_linear_init(self):
        """Post-materialization reset equals the next fresh Linear init (RNG continuation)."""
        torch.manual_seed(31)
        target = torch.nn.LazyLinear(5)
        target.initialize_parameters(torch.randn(2, 7))  # consumes D1
        reference = torch.nn.Linear(7, 5)  # consumes D2 from the same stream
        torch.manual_seed(31)
        other = torch.nn.LazyLinear(5)
        other.initialize_parameters(torch.randn(2, 7))  # consumes D1 again
        other.reset_parameters()  # must draw D2
        self.assertTrue(torch.equal(other.weight, reference.weight))
        self.assertTrue(torch.equal(other.bias, reference.bias))
        self.assertFalse(torch.equal(target.weight, other.weight))

    def test_reset_parameters_after_initialization(self):
        """After materialization, reset restores the Linear init distribution."""
        m = torch.nn.LazyLinear(5)
        m.initialize_parameters(torch.randn(2, 7))
        with torch.no_grad():
            m.weight.fill_(999.0)
            m.bias.fill_(999.0)
        m.reset_parameters()
        bound = 1.0 / math.sqrt(7)
        self.assertTrue(bool((m.weight.abs() <= bound).all()))
        self.assertTrue(bool((m.bias.abs() <= bound).all()))
        self.assertTrue(bool((m.weight != 999.0).all()))

    def test_reset_parameters_on_device(self):
        """Device-materialized module keeps device and objects across resets."""
        m = torch.nn.LazyLinear(5, device=self.device)
        m.initialize_parameters(torch.randn(2, 7, device=self.device))
        weight_ref = m.weight
        bias_ref = m.bias
        with torch.no_grad():
            m.weight.fill_(999.0)
            m.bias.fill_(999.0)
        m.reset_parameters()
        self.assertIs(m.weight, weight_ref)
        self.assertIs(m.bias, bias_ref)
        self.assertEqual(m.weight.device.type, self.device_name)
        self.assertEqual(m.bias.device.type, self.device_name)
        bound = 1.0 / math.sqrt(7)
        self.assertTrue(bool((m.weight.abs() <= bound).all()))
        self.assertTrue(bool((m.bias.abs() <= bound).all()))
        self.assertEqual(tuple(m.weight.shape), (5, 7))

    def test_reset_parameters_forward_after_reset_matches_golden(self):
        """Forward after reset matches the CPU golden computed from the same weights."""
        m = torch.nn.LazyLinear(5, device=self.device)
        m.initialize_parameters(torch.randn(2, 7, device=self.device))
        m.reset_parameters()
        x = torch.randn(2, 7, device=self.device)
        out = m(x)
        golden = torch.nn.functional.linear(
            x.detach().cpu(), m.weight.detach().cpu(), m.bias.detach().cpu())
        torch.testing.assert_close(out.detach().cpu(), golden)

    def test_reset_parameters_after_full_forward(self):
        """After a full forward the lazy module became Linear; reset still works."""
        m = torch.nn.LazyLinear(5, device=self.device)
        m(torch.randn(2, 7, device=self.device))
        self.assertIs(type(m), torch.nn.Linear)
        with torch.no_grad():
            m.weight.fill_(999.0)
            m.bias.fill_(999.0)
        m.reset_parameters()
        bound = 1.0 / math.sqrt(7)
        self.assertTrue(bool((m.weight.abs() <= bound).all()))
        self.assertTrue(bool((m.bias.abs() <= bound).all()))
        self.assertTrue(bool((m.weight != 999.0).all()))


if __name__ == "__main__":
    run_tests()
