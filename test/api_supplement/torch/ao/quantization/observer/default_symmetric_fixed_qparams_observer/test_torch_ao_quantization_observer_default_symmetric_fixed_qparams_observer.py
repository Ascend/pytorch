# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.observer.default_symmetric_fixed_qparams_observer 工厂接口正确性
API 名称：torch.ao.quantization.observer.default_symmetric_fixed_qparams_observer
API 签名：default_symmetric_fixed_qparams_observer = default_fixed_qparams_range_neg1to1_observer
          # 即 FixedQParamsObserver.with_args(scale=2.0/256.0, zero_point=128, ...)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | 工厂无可选空参数，调用即产出配置固定的观察者 | 已覆盖：test_factory_builds_fixed_qparams_observer |
| 枚举选项 | dtype 固定为 torch.quint8 | 已覆盖：test_observer_dtype_and_range |
| 参数类型 | 工厂返回 _PartialWrapper，调用返回 FixedQParamsObserver | 已覆盖：test_factory_returns_partial_wrapper |
| 传参与不传参 | 工厂无参调用使用 with_args 固化参数 | 已覆盖：test_factory_builds_fixed_qparams_observer |
| 等价类/边界值 | 与 range_neg1to1 为同一对象，scale=2/256、zero_point=128 | 已覆盖：test_alias_identity / test_observer_fixed_scale_and_zero_point |
| 正常传参场景 | calculate_qparams 与 forward 语义 | 已覆盖：test_calculate_qparams_matches_buffers / test_forward_is_identity |
| 异常传参场景 | 工厂为固化 partial，无稳定参数校验异常 | 未覆盖，原因见下 |
| 混合设备类型 | 固定 qparams 观察者不涉及多 Tensor 异构设备输入 | 不适用 |

未覆盖项及原因：
- 异常传参场景：default_symmetric_fixed_qparams_observer 是 range_neg1to1 的等价别名（同一 partial），
  不接受构造参数，没有稳定异常路径可用于 assertRaises。
- 混合设备类型：该工厂只产出配置固定的观察者，不接收张量，无法构造异构设备输入。

注：default_symmetric_fixed_qparams_observer 为向后兼容别名，无 CPU/NPU 数值对照路径，
     故不做 CPU/NPU Golden 对照，仅验证别名同一性、scale/zero_point 与 forward 语义。
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

from torch.ao.quantization.observer import (
    _PartialWrapper,
    default_fixed_qparams_range_neg1to1_observer,
    default_symmetric_fixed_qparams_observer,
    FixedQParamsObserver,
)

_EXPECTED_SCALE = 2.0 / 256.0
_EXPECTED_ZERO_POINT = 128


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestTorchAoQuantizationObserverDefaultSymmetricFixedQparamsUt(TestCase):
    """Functional tests for default_symmetric_fixed_qparams_observer."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_factory_returns_partial_wrapper(self):
        """The factory is a with_args partial wrapper, not a bare observer class."""
        self.assertIsInstance(default_symmetric_fixed_qparams_observer, _PartialWrapper)

    def test_alias_identity(self):
        """The symmetric alias is the very same object as the neg1to1 range factory."""
        self.assertIs(default_symmetric_fixed_qparams_observer, default_fixed_qparams_range_neg1to1_observer)

    def test_factory_builds_fixed_qparams_observer(self):
        """Calling the factory yields a FixedQParamsObserver instance."""
        self.assertIsInstance(default_symmetric_fixed_qparams_observer(), FixedQParamsObserver)

    def test_observer_dtype_and_range(self):
        """The observer is configured for the quint8 [0, 255] range."""
        obs = default_symmetric_fixed_qparams_observer()
        self.assertEqual(obs.dtype, torch.quint8)
        self.assertEqual(obs.quant_min, 0)
        self.assertEqual(obs.quant_max, 255)

    def test_observer_fixed_scale_and_zero_point(self):
        """The fixed scale is 2/256 and zero_point is the symmetric midpoint 128."""
        obs = default_symmetric_fixed_qparams_observer()
        torch.testing.assert_close(obs.scale, torch.tensor([_EXPECTED_SCALE], dtype=torch.float))
        self.assertEqual(int(obs.zero_point[0]), _EXPECTED_ZERO_POINT)

    def test_calculate_qparams_matches_buffers(self):
        """calculate_qparams returns the frozen scale/zero_point buffers unchanged."""
        obs = default_symmetric_fixed_qparams_observer()
        scale, zero_point = obs.calculate_qparams()
        torch.testing.assert_close(scale, obs.scale)
        self.assertEqual(int(zero_point[0]), _EXPECTED_ZERO_POINT)

    def test_forward_is_identity(self):
        """The fixed-qparams observer passes its input through untouched."""
        obs = default_symmetric_fixed_qparams_observer()
        x = torch.randn(2, 3)
        self.assertIs(obs(x), x)


if __name__ == "__main__":
    run_tests()
