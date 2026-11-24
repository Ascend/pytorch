# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.nn.quantized.dynamic.modules.rnn.LSTMCell 接口功能正确性
API 名称：torch.ao.nn.quantized.dynamic.modules.rnn.LSTMCell
API 签名：LSTMCell(input_size, hidden_size, bias=True, dtype=torch.qint8)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | hx 为空(None)与非空两种前向 | 已覆盖：test_lstmcell_forward_returns_hidden_cell、test_lstmcell_forward_with_hx |
| 枚举选项 | 权重 dtype 枚举 qint8 | 已覆盖：test_lstmcell_construct_attributes |
| 参数类型 | input 为 2D Tensor，hx 为 (h, c) 元组 | 已覆盖：test_lstmcell_forward_with_hx |
| 传参与不传参 | bias 默认 True 与显式 False | 已覆盖：test_lstmcell_bias_disabled |
| 等价类/边界值 | 单样本 batch=1 输入 | 已覆盖：test_lstmcell_forward_single_sample |
| 正常传参场景 | 返回 (h, c) 且 shape 正确 | 已覆盖：test_lstmcell_forward_returns_hidden_cell |
| 异常传参场景 | 尺寸不匹配报错 | 已覆盖：lstmcell_check_forward_input_mismatch、lstmcell_check_forward_hidden_mismatch |
| 混合设备类型 | 多 Tensor 异构设备输入 | 未覆盖，原因见下 |

未覆盖项及原因：
- 混合设备类型：动态量化 LSTMCell 仅由 CPU 量化后端（onednn/fbgemm/qnnpack）提供内核，
  NPU/CUDA 上无对应执行路径，无法构造异构设备输入。
- float16 权重：fp16 动态内核仅 fbgemm/onednn 提供，qnnpack 平台不可用，故未覆盖。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注：动态量化 RNNCell 只在 CPU 量化后端执行，无 NPU/CUDA 数值对照路径，
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
from torch.ao.nn.quantized.dynamic.modules.rnn import LSTMCell

_ENGINE_CANDIDATES = ("onednn", "fbgemm", "qnnpack")
_INPUT_SIZE = 3
_HIDDEN_SIZE = 7
_BATCH = 4
_WEIGHT_QPARAMS = {
    "qscheme": torch.per_tensor_affine,
    "dtype": torch.qint8,
    "scale": 0.1,
    "zero_point": 0,
}


def _use_quantized_engine(probe=None):
    """Select a CPU quantized engine able to run the target op.

    The dynamic quantized LSTMCell only provides kernels on the CPU quantized
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
    LSTMCell(_INPUT_SIZE, _HIDDEN_SIZE)(torch.randn(_BATCH, _INPUT_SIZE))


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestLstmcellUt(TestCase):
    """Functional tests for the dynamic quantized LSTMCell module."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        _use_quantized_engine(_probe)

    def test_lstmcell_construct_attributes(self):
        cell = LSTMCell(_INPUT_SIZE, _HIDDEN_SIZE)
        self.assertEqual(cell.input_size, _INPUT_SIZE)
        self.assertEqual(cell.hidden_size, _HIDDEN_SIZE)
        self.assertTrue(cell.bias)
        self.assertEqual(cell.weight_dtype, torch.qint8)

    def test_lstmcell_bias_disabled(self):
        cell = LSTMCell(_INPUT_SIZE, _HIDDEN_SIZE, bias=False)
        self.assertFalse(cell.bias)
        self.assertIsNone(cell.bias_ih)
        self.assertIsNone(cell.bias_hh)

    def test_lstmcell_forward_returns_hidden_cell(self):
        cell = LSTMCell(_INPUT_SIZE, _HIDDEN_SIZE)
        h, c = cell(torch.randn(_BATCH, _INPUT_SIZE))
        self.assertEqual(tuple(h.shape), (_BATCH, _HIDDEN_SIZE))
        self.assertEqual(tuple(c.shape), (_BATCH, _HIDDEN_SIZE))

    def test_lstmcell_forward_with_hx(self):
        cell = LSTMCell(_INPUT_SIZE, _HIDDEN_SIZE)
        x = torch.randn(_BATCH, _INPUT_SIZE)
        h0 = torch.randn(_BATCH, _HIDDEN_SIZE)
        c0 = torch.randn(_BATCH, _HIDDEN_SIZE)
        h, c = cell(x, (h0, c0))
        self.assertEqual(tuple(h.shape), (_BATCH, _HIDDEN_SIZE))
        self.assertEqual(tuple(c.shape), (_BATCH, _HIDDEN_SIZE))

    def test_lstmcell_forward_single_sample(self):
        cell = LSTMCell(_INPUT_SIZE, _HIDDEN_SIZE)
        h, c = cell(torch.randn(1, _INPUT_SIZE))
        self.assertEqual(tuple(h.shape), (1, _HIDDEN_SIZE))

    def test_lstmcell_check_forward_input_mismatch(self):
        cell = LSTMCell(_INPUT_SIZE, _HIDDEN_SIZE)
        with self.assertRaises(RuntimeError):
            cell(torch.randn(_BATCH, _INPUT_SIZE + 1))

    def test_lstmcell_check_forward_hidden_mismatch(self):
        cell = LSTMCell(_INPUT_SIZE, _HIDDEN_SIZE)
        x = torch.randn(_BATCH, _INPUT_SIZE)
        h0 = torch.randn(_BATCH, _HIDDEN_SIZE + 1)
        c0 = torch.randn(_BATCH, _HIDDEN_SIZE)
        with self.assertRaises(RuntimeError):
            cell(x, (h0, c0))

    def test_lstmcell_from_float_returns_lstmcell(self):
        float_mod = torch.nn.LSTMCell(_INPUT_SIZE, _HIDDEN_SIZE)
        float_mod.qconfig = torch.ao.quantization.default_dynamic_qconfig
        cell = LSTMCell.from_float(float_mod)
        self.assertIsInstance(cell, LSTMCell)
        self.assertEqual(cell.weight_dtype, torch.qint8)
        h, c = cell(torch.randn(_BATCH, _INPUT_SIZE))
        self.assertEqual(tuple(h.shape), (_BATCH, _HIDDEN_SIZE))

    def test_lstmcell_from_float_rejects_unsupported_type(self):
        with self.assertRaises(AssertionError):
            LSTMCell.from_float(torch.nn.Linear(_INPUT_SIZE, _HIDDEN_SIZE))

    def test_lstmcell_from_reference_returns_lstmcell(self):
        cell = LSTMCell.from_reference(_build_reference_lstmcell())
        self.assertIsInstance(cell, LSTMCell)
        h, c = cell(torch.randn(_BATCH, _INPUT_SIZE))
        self.assertEqual(tuple(h.shape), (_BATCH, _HIDDEN_SIZE))

    def test_lstmcell_get_name(self):
        cell = LSTMCell(_INPUT_SIZE, _HIDDEN_SIZE)
        self.assertEqual(cell._get_name(), "DynamicQuantizedLSTMCell")


if __name__ == "__main__":
    run_tests()
