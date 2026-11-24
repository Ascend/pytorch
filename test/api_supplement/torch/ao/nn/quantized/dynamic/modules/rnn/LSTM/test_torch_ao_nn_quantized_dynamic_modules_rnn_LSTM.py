# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.nn.quantized.dynamic.modules.rnn.LSTM 接口功能正确性
API 名称：torch.ao.nn.quantized.dynamic.modules.rnn.LSTM
API 签名：LSTM(input_size, hidden_size, num_layers=1, bias=True, batch_first=False,
              dropout=0.0, bidirectional=False, dtype=torch.qint8)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | hx 为空(None)与非空两种前向 | 已覆盖：test_lstm_forward_shape、test_lstm_forward_with_hx |
| 枚举选项 | dropout/bidirectional/num_layers 组合 | 已覆盖：test_lstm_forward_multi_layer_bidirectional |
| 参数类型 | 输入为 Tensor 或 PackedSequence | 已覆盖：test_lstm_forward_packed_sequence |
| 传参与不传参 | batch_first 默认 False 与显式 True | 已覆盖：test_lstm_forward_batch_first |
| 等价类/边界值 | 单层单向与多层双向 | 已覆盖：test_lstm_forward_shape、test_lstm_forward_multi_layer_bidirectional |
| 正常传参场景 | 输出 (output, (h, c)) shape 正确 | 已覆盖：test_lstm_forward_shape |
| 异常传参场景 | 输入维度/hidden 尺寸不匹配触发 RuntimeError | 已覆盖：test_lstm_check_input_bad_dim、test_lstm_check_hidden_bad_size |
| 混合设备类型 | 多 Tensor 异构设备输入 | 未覆盖，原因见下 |

未覆盖项及原因：
- 混合设备类型：动态量化 LSTM 仅由 CPU 量化后端（onednn/fbgemm/qnnpack）提供内核，
  NPU/CUDA 上无对应执行路径，无法构造异构设备输入。
- float16 权重：fp16 动态内核仅 fbgemm/onednn 提供，qnnpack 平台不可用，故未覆盖。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注：动态量化 LSTM 只在 CPU 量化后端执行，无 NPU/CUDA 数值对照路径，
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
from torch.ao.nn.quantized.dynamic.modules.rnn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence

_ENGINE_CANDIDATES = ("onednn", "fbgemm", "qnnpack")
_INPUT_SIZE = 3
_HIDDEN_SIZE = 7
_BATCH = 2
_SEQ_LEN = 4
_WEIGHT_QPARAMS = {
    "qscheme": torch.per_tensor_affine,
    "dtype": torch.qint8,
    "scale": 0.1,
    "zero_point": 0,
}


def _use_quantized_engine(probe=None):
    """Select a CPU quantized engine able to run the target op.

    The dynamic quantized LSTM only provides kernels on the CPU quantized
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


def _build_reference_lstm():
    weight_qparams_dict = {"is_decomposed": False, "weight_ih_l0": _WEIGHT_QPARAMS, "weight_hh_l0": _WEIGHT_QPARAMS}
    return torch.ao.nn.quantized.reference.LSTM(_INPUT_SIZE, _HIDDEN_SIZE, 1, weight_qparams_dict=weight_qparams_dict)


def _probe():
    lstm = LSTM(_INPUT_SIZE, _HIDDEN_SIZE, 1)
    lstm(torch.randn(_SEQ_LEN, _BATCH, _INPUT_SIZE))


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestLstmUt(TestCase):
    """Functional tests for the dynamic quantized LSTM module."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        _use_quantized_engine(_probe)

    def test_lstm_construct_attributes(self):
        lstm = LSTM(_INPUT_SIZE, _HIDDEN_SIZE, 2)
        self.assertEqual(lstm.input_size, _INPUT_SIZE)
        self.assertEqual(lstm.hidden_size, _HIDDEN_SIZE)
        self.assertEqual(lstm.num_layers, 2)
        self.assertEqual(lstm.mode, "LSTM")
        self.assertEqual(lstm.dtype, torch.qint8)

    def test_lstm_forward_shape(self):
        lstm = LSTM(_INPUT_SIZE, _HIDDEN_SIZE, 1)
        out, (hn, cn) = lstm(torch.randn(_SEQ_LEN, _BATCH, _INPUT_SIZE))
        self.assertEqual(tuple(out.shape), (_SEQ_LEN, _BATCH, _HIDDEN_SIZE))
        self.assertEqual(tuple(hn.shape), (1, _BATCH, _HIDDEN_SIZE))
        self.assertEqual(tuple(cn.shape), (1, _BATCH, _HIDDEN_SIZE))

    def test_lstm_forward_with_hx(self):
        lstm = LSTM(_INPUT_SIZE, _HIDDEN_SIZE, 1)
        h0 = torch.randn(1, _BATCH, _HIDDEN_SIZE)
        c0 = torch.randn(1, _BATCH, _HIDDEN_SIZE)
        out, (hn, cn) = lstm(torch.randn(_SEQ_LEN, _BATCH, _INPUT_SIZE), (h0, c0))
        self.assertEqual(tuple(out.shape), (_SEQ_LEN, _BATCH, _HIDDEN_SIZE))
        self.assertEqual(tuple(hn.shape), (1, _BATCH, _HIDDEN_SIZE))

    def test_lstm_forward_multi_layer_bidirectional(self):
        lstm = LSTM(_INPUT_SIZE, _HIDDEN_SIZE, 2, bidirectional=True)
        num_directions = 2
        out, (hn, cn) = lstm(torch.randn(_SEQ_LEN, _BATCH, _INPUT_SIZE))
        self.assertEqual(tuple(out.shape), (_SEQ_LEN, _BATCH, num_directions * _HIDDEN_SIZE))
        self.assertEqual(tuple(hn.shape), (2 * num_directions, _BATCH, _HIDDEN_SIZE))
        self.assertEqual(len(lstm._all_weight_values), 2 * num_directions)

    def test_lstm_forward_batch_first(self):
        lstm = LSTM(_INPUT_SIZE, _HIDDEN_SIZE, 1, batch_first=True)
        out, (hn, cn) = lstm(torch.randn(_BATCH, _SEQ_LEN, _INPUT_SIZE))
        self.assertEqual(tuple(out.shape), (_BATCH, _SEQ_LEN, _HIDDEN_SIZE))

    def test_lstm_forward_packed_sequence(self):
        lstm = LSTM(_INPUT_SIZE, _HIDDEN_SIZE, 1)
        seq = torch.randn(_SEQ_LEN, _BATCH, _INPUT_SIZE)
        packed = pack_padded_sequence(seq, torch.tensor([_SEQ_LEN, _SEQ_LEN - 1]))
        out, (hn, cn) = lstm(packed)
        self.assertEqual(out.data.shape[1], _HIDDEN_SIZE)
        self.assertEqual(tuple(hn.shape), (1, _BATCH, _HIDDEN_SIZE))

    def test_lstm_check_input_bad_dim(self):
        lstm = LSTM(_INPUT_SIZE, _HIDDEN_SIZE, 1)
        with self.assertRaises(RuntimeError):
            lstm(torch.randn(_BATCH, _INPUT_SIZE))

    def test_lstm_check_hidden_bad_size(self):
        lstm = LSTM(_INPUT_SIZE, _HIDDEN_SIZE, 1)
        h0 = torch.randn(1, _BATCH, _HIDDEN_SIZE + 1)
        c0 = torch.randn(1, _BATCH, _HIDDEN_SIZE)
        with self.assertRaises(RuntimeError):
            lstm(torch.randn(_SEQ_LEN, _BATCH, _INPUT_SIZE), (h0, c0))

    def test_lstm_from_float_returns_lstm(self):
        float_mod = torch.nn.LSTM(_INPUT_SIZE, _HIDDEN_SIZE, 2)
        float_mod.qconfig = torch.ao.quantization.default_dynamic_qconfig
        lstm = LSTM.from_float(float_mod)
        self.assertIsInstance(lstm, LSTM)
        self.assertEqual(lstm.num_layers, 2)
        self.assertEqual(lstm.dtype, torch.qint8)
        self.assertEqual(len(lstm._all_weight_values), 2)

    def test_lstm_from_float_rejects_unsupported_type(self):
        with self.assertRaises(AssertionError):
            LSTM.from_float(torch.nn.RNN(_INPUT_SIZE, _HIDDEN_SIZE))

    def test_lstm_from_float_requires_qconfig(self):
        with self.assertRaises(AssertionError):
            LSTM.from_float(torch.nn.LSTM(_INPUT_SIZE, _HIDDEN_SIZE, 1))

    def test_lstm_from_reference_returns_lstm(self):
        lstm = LSTM.from_reference(_build_reference_lstm())
        self.assertIsInstance(lstm, LSTM)
        self.assertEqual(lstm.dtype, torch.qint8)
        out, (hn, cn) = lstm(torch.randn(_SEQ_LEN, _BATCH, _INPUT_SIZE))
        self.assertEqual(tuple(out.shape), (_SEQ_LEN, _BATCH, _HIDDEN_SIZE))

    def test_lstm_from_reference_missing_attr(self):
        with self.assertRaises(AssertionError):
            LSTM.from_reference(torch.nn.LSTM(_INPUT_SIZE, _HIDDEN_SIZE, 1))

    def test_lstm_get_name(self):
        lstm = LSTM(_INPUT_SIZE, _HIDDEN_SIZE, 1)
        self.assertEqual(lstm._get_name(), "DynamicQuantizedLSTM")


if __name__ == "__main__":
    run_tests()
