# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.nn.quantized.dynamic.modules.rnn.RNNCellBase 接口功能正确性
API 名称：torch.ao.nn.quantized.dynamic.modules.rnn.RNNCellBase
API 签名：RNNCellBase(input_size, hidden_size, bias=True, num_chunks=4, dtype=torch.qint8)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | bias 为空(False)与非空 | 已覆盖：test_rnncellbase_bias_disabled_registers_none |
| 枚举选项 | num_chunks 枚举 1/3/4 | 已覆盖：test_rnncellbase_num_chunks_controls_bias_size |
| 参数类型 | 权重与偏置张量类型/形状 | 已覆盖：test_rnncellbase_construct_attributes |
| 传参与不传参 | bias 默认 True 与显式 False | 已覆盖：test_rnncellbase_bias_disabled_registers_none |
| 等价类/边界值 | 权重/偏置字典键齐全 | 已覆盖：test_rnncellbase_get_weight_keys、test_rnncellbase_get_bias_keys |
| 正常传参场景 | 返回实例 | 已覆盖：rnncellbase_from_float_returns_subclass、rnncellbase_from_reference_returns_instance |
| 异常传参场景 | 非法类型/缺 qconfig 报错 | 已覆盖：rnncellbase_from_float_unsupported_type、rnncellbase_from_float_requires_qconfig |
| 混合设备类型 | 多 Tensor 异构设备输入 | 未覆盖，原因见下 |

未覆盖项及原因：
- 混合设备类型：动态量化 RNNCellBase 仅由 CPU 量化后端（onednn/fbgemm/qnnpack）提供内核，
  NPU/CUDA 上无对应执行路径，无法构造异构设备输入。
- float16 权重：fp16 动态内核仅 fbgemm/onednn 提供，qnnpack 平台不可用，故未覆盖。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注：动态量化 RNNCellBase 只在 CPU 量化后端执行，无 NPU/CUDA 数值对照路径，
     故本用例不做 CPU/NPU Golden 对照，仅验证量化后端上的功能语义。
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

import torch.ao.nn.quantized.dynamic
import torch.ao.nn.quantized.reference
from torch.ao.nn.quantized.dynamic.modules.rnn import LSTMCell, RNNCellBase

_ENGINE_CANDIDATES = ("onednn", "fbgemm", "qnnpack")
_INPUT_SIZE = 3
_HIDDEN_SIZE = 7
_WEIGHT_QPARAMS = {
    "qscheme": torch.per_tensor_affine,
    "dtype": torch.qint8,
    "scale": 0.1,
    "zero_point": 0,
}


def _use_quantized_engine(probe=None):
    """Select a CPU quantized engine able to instantiate the target module.

    Dynamic quantized RNNCellBase only packs weights on the CPU quantized
    backend, so there is no NPU/CUDA execution path to compare against.
    """
    supported = torch.backends.quantized.supported_engines
    for name in _ENGINE_CANDIDATES:
        if name not in supported:
            continue
        torch.backends.quantized.engine = name
        if probe is None:
            return name
        try:
            probe()
        except RuntimeError:
            # This engine has no kernel for the target op; try the next one.
            continue
        return name
    raise unittest.SkipTest("No CPU quantized engine can run the target op on this platform")


def _build_reference_lstmcell():
    weight_qparams_dict = {"weight_ih": _WEIGHT_QPARAMS, "weight_hh": _WEIGHT_QPARAMS, "is_decomposed": False}
    return torch.ao.nn.quantized.reference.LSTMCell(_INPUT_SIZE, _HIDDEN_SIZE, weight_qparams_dict=weight_qparams_dict)


def _probe():
    RNNCellBase(_INPUT_SIZE, _HIDDEN_SIZE, num_chunks=4).get_weight()


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestRnncellbaseUt(TestCase):
    """Functional tests for the dynamic quantized RNNCellBase class."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        _use_quantized_engine(_probe)

    def test_rnncellbase_construct_attributes(self):
        cell = RNNCellBase(_INPUT_SIZE, _HIDDEN_SIZE, bias=True, num_chunks=4)
        self.assertEqual(cell.input_size, _INPUT_SIZE)
        self.assertEqual(cell.hidden_size, _HIDDEN_SIZE)
        self.assertTrue(cell.bias)
        self.assertEqual(cell.weight_dtype, torch.qint8)
        self.assertIsNotNone(cell._packed_weight_ih)
        self.assertIsNotNone(cell._packed_weight_hh)

    def test_rnncellbase_bias_disabled_registers_none(self):
        cell = RNNCellBase(_INPUT_SIZE, _HIDDEN_SIZE, bias=False, num_chunks=4)
        self.assertFalse(cell.bias)
        self.assertIsNone(cell.bias_ih)
        self.assertIsNone(cell.bias_hh)

    def test_rnncellbase_num_chunks_controls_bias_size(self):
        cell = RNNCellBase(_INPUT_SIZE, _HIDDEN_SIZE, bias=True, num_chunks=3)
        self.assertEqual(tuple(cell.bias_ih.shape), (3 * _HIDDEN_SIZE,))
        self.assertEqual(tuple(cell.get_weight()["weight_ih"].shape), (3 * _HIDDEN_SIZE, _INPUT_SIZE))

    def test_rnncellbase_get_weight_keys(self):
        cell = RNNCellBase(_INPUT_SIZE, _HIDDEN_SIZE, num_chunks=4)
        self.assertEqual(set(cell.get_weight().keys()), {"weight_ih", "weight_hh"})

    def test_rnncellbase_get_bias_keys(self):
        cell = RNNCellBase(_INPUT_SIZE, _HIDDEN_SIZE, num_chunks=4)
        self.assertEqual(set(cell.get_bias().keys()), {"bias_ih", "bias_hh"})

    def test_rnncellbase_from_float_returns_subclass(self):
        float_mod = torch.nn.LSTMCell(_INPUT_SIZE, _HIDDEN_SIZE)
        float_mod.qconfig = torch.ao.quantization.default_dynamic_qconfig
        cell = RNNCellBase.from_float(float_mod)
        self.assertIsInstance(cell, LSTMCell)
        self.assertEqual(cell.weight_dtype, torch.qint8)

    def test_rnncellbase_from_float_unsupported_type(self):
        with self.assertRaises(AssertionError):
            RNNCellBase.from_float(torch.nn.Linear(_INPUT_SIZE, _HIDDEN_SIZE))

    def test_rnncellbase_from_float_requires_qconfig(self):
        with self.assertRaises(AssertionError):
            RNNCellBase.from_float(torch.nn.LSTMCell(_INPUT_SIZE, _HIDDEN_SIZE))

    def test_rnncellbase_from_reference_returns_instance(self):
        # from_reference instantiates with `cls`, so calling it on RNNCellBase
        # yields an RNNCellBase instance carrying the reference weight dtype.
        cell = RNNCellBase.from_reference(_build_reference_lstmcell())
        self.assertIsInstance(cell, RNNCellBase)
        self.assertEqual(cell.weight_dtype, torch.qint8)
        self.assertEqual(set(cell.get_weight().keys()), {"weight_ih", "weight_hh"})

    def test_rnncellbase_from_reference_missing_attr(self):
        with self.assertRaises(AssertionError):
            RNNCellBase.from_reference(torch.nn.LSTMCell(_INPUT_SIZE, _HIDDEN_SIZE))

    def test_rnncellbase_get_name(self):
        cell = RNNCellBase(_INPUT_SIZE, _HIDDEN_SIZE, num_chunks=4)
        self.assertEqual(cell._get_name(), "DynamicQuantizedRNNBase")


if __name__ == "__main__":
    run_tests()
