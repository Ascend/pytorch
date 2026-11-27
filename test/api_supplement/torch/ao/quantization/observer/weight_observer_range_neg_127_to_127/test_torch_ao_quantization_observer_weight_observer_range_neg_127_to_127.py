# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.observer.weight_observer_range_neg_127_to_127 工厂接口正确性
API 名称：torch.ao.quantization.observer.weight_observer_range_neg_127_to_127
API 签名：MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric,
          quant_min=-127, quant_max=127, eps=2**-12)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | forward 前 min/max 缓冲为空、forward 后非空 | 已覆盖：test_forward_then_calculate_qparams |
| 枚举选项 | qscheme 固定为 per_tensor_symmetric，dtype 固定为 qint8 | 已覆盖：test_observer_dtype_and_qscheme |
| 参数类型 | 工厂返回 _PartialWrapper，调用返回 MinMaxObserver | 已覆盖：test_factory_returns_partial_wrapper |
| 传参与不传参 | 工厂无参调用使用 with_args 固化参数 | 已覆盖：test_factory_builds_minmax_observer |
| 等价类/边界值 | [-127, 127] 区间（排除 -128）的对称量化 | 已覆盖：test_scale_uses_127_range / test_observer_quant_range |
| 正常传参场景 | forward 后 scale>0、symmetric zero_point=0 | 已覆盖：test_forward_then_calculate_qparams |
| 异常传参场景 | qscheme/dtype 由工厂固化，无稳定异常路径 | 未覆盖，原因见下 |
| 混合设备类型 | per-tensor 观察者不涉及多 Tensor 异构设备输入 | 不适用 |

未覆盖项及原因：
- 异常传参场景：weight_observer_range_neg_127_to_127 是 with_args 固化 partial，不接受构造参数；
  dtype/qscheme/quant_min/quant_max 均已固定，无稳定异常路径可用于 assertRaises。
- 混合设备类型：该工厂只产出观察者，不接收张量，无法构造异构设备输入。

注：该权重观察者只在 CPU 量化后端上有对应内核，无 NPU/CUDA 执行路径，
     故不做 CPU/NPU Golden 对照，仅验证 dtype/qscheme/量化区间与对称 scale 语义。
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
    MinMaxObserver,
    weight_observer_range_neg_127_to_127,
)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestTorchAoQuantizationObserverWeightObserverRangeNeg127To127Ut(TestCase):
    """Functional tests for weight_observer_range_neg_127_to_127."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_factory_returns_partial_wrapper(self):
        """The factory is a with_args partial wrapper, not a bare observer class."""
        self.assertIsInstance(weight_observer_range_neg_127_to_127, _PartialWrapper)

    def test_factory_builds_minmax_observer(self):
        """Calling the factory yields a MinMaxObserver instance."""
        self.assertIsInstance(weight_observer_range_neg_127_to_127(), MinMaxObserver)

    def test_observer_dtype_and_qscheme(self):
        """The observer is a per-tensor symmetric qint8 weight observer."""
        obs = weight_observer_range_neg_127_to_127()
        self.assertEqual(obs.dtype, torch.qint8)
        self.assertEqual(obs.qscheme, torch.per_tensor_symmetric)

    def test_observer_quant_range(self):
        """The quantized range is [-127, 127], excluding -128, with eps=2**-12."""
        obs = weight_observer_range_neg_127_to_127()
        self.assertEqual(obs.quant_min, -127)
        self.assertEqual(obs.quant_max, 127)
        self.assertEqual(float(obs.eps), 2 ** -12)

    def test_forward_then_calculate_qparams(self):
        """After observing a tensor the symmetric zero_point is 0 and scale is positive."""
        obs = weight_observer_range_neg_127_to_127()
        obs(torch.tensor([[-3.0, 5.0, 1.0]]))
        scale, zero_point = obs.calculate_qparams()
        self.assertTrue(torch.all(scale > 0))
        self.assertEqual(int(zero_point[0]), 0)

    def test_scale_uses_127_range(self):
        """max_abs 5 maps to scale 5/127, proving the -128 level is excluded."""
        obs = weight_observer_range_neg_127_to_127()
        obs(torch.tensor([[-3.0, 5.0, 1.0]]))
        scale, _ = obs.calculate_qparams()
        # If the implementation used [-128, 127] the scale would be 5/127.5 instead.
        torch.testing.assert_close(scale, torch.tensor([5.0 / 127.0]))


if __name__ == "__main__":
    run_tests()
